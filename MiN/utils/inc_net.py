import copy
import torch
from torch import nn
from torch.nn import functional as F
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast 
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear

class RandomBuffer(torch.nn.Linear):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features, self.out_features = in_features, buffer_size
        factory_kwargs = {"device": device, "dtype": torch.float32}
        self.register_buffer("weight", torch.empty((self.in_features, self.out_features), **factory_kwargs))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Ép X về device/type của weight (GPU/Float32)
        return F.relu(X.to(device=self.weight.device, dtype=self.weight.dtype) @ self.weight)

class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args, self.device = args, args['device']
        self.backbone = get_pretrained_backbone(args).to(self.device)
        self.gamma, self.buffer_size = args['gamma'], args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        self.buffer = RandomBuffer(self.feature_dim, self.buffer_size, self.device)
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), device=self.device, dtype=torch.float32))

        # DPCR Storage 100% trên GPU
        self._compressed_stats = {} 
        self._mu_list = {}          
        self._class_counts = {}     
        self.temp_phi, self.temp_mu, self.temp_count = {}, {}, {}

        self.normal_fc = None
        self.cur_task, self.known_class = -1, 0

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=(self.cur_task==0)).to(self.device)
        if self.normal_fc is not None:
            new_fc.weight.data[:self.normal_fc.out_features] = self.normal_fc.weight.data
            if new_fc.bias is not None and self.normal_fc.bias is not None:
                new_fc.bias.data[:self.normal_fc.out_features] = self.normal_fc.bias.data
        self.normal_fc = new_fc

    @torch.no_grad()
    def accumulate_stats(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.eval()
        # Đưa input lên GPU và ép float32 ngay lập tức
        X, Y = X.to(self.device, dtype=torch.float32), Y.to(self.device, dtype=torch.float32)
        
        with autocast('cuda', enabled=False):
            feat = self.buffer(self.backbone(X))
        
        labels = torch.argmax(Y, dim=1)
        for i in range(feat.shape[0]):
            label = labels[i].item()
            f = feat[i:i+1]
            if label not in self.temp_phi:
                self.temp_phi[label] = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
                self.temp_mu[label] = torch.zeros(self.buffer_size, device=self.device)
                self.temp_count[label] = 0
            self.temp_phi[label] += (f.t() @ f)
            self.temp_mu[label] += f.squeeze(0)
            self.temp_count[label] += 1

    @torch.no_grad()
    def compress_stats(self):
        """Nén trực tiếp trên GPU"""
        RANK = 256
        for label in sorted(list(self.temp_phi.keys())):
            raw_phi = self.temp_phi[label]
            # Tính eigh thẳng trên GPU
            S, V = torch.linalg.eigh(raw_phi)
            
            S_top, V_top = (S[-RANK:], V[:, -RANK:]) if S.shape[0] > RANK else (S, V)
            # Lưu kết quả cũng trên GPU luôn
            self._compressed_stats[label] = (V_top, S_top)
            self._mu_list[label] = (self.temp_mu[label] / self.temp_count[label])
            self._class_counts[label] = self.temp_count[label]
            
            del self.temp_phi[label], self.temp_mu[label], raw_phi
        torch.cuda.empty_cache()

    @torch.no_grad()
    def solve_dpcr(self, P_drift=None, boundary=0, init_mode=False):
        """Giải hệ phương trình 100% GPU"""
        num_total = max(list(self._compressed_stats.keys()) + list(self.temp_phi.keys())) + 1
        if num_total > self.weight.shape[1]:
            new_w = torch.zeros((self.buffer_size, num_total), device=self.device)
            new_w[:, :self.weight.shape[1]] = self.weight
            self.register_buffer("weight", new_w)

        total_phi = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
        total_q = torch.zeros(self.buffer_size, num_total, device=self.device)

        for c, (V, S) in self._compressed_stats.items():
            phi_c = (V @ torch.diag(S)) @ V.t()
            mu_c = self._mu_list[c]
            if P_drift is not None and c < boundary:
                P_drift = P_drift.to(self.device)
                P_cs = V @ V.t()
                phi_c = P_drift.t() @ P_cs.t() @ phi_c @ P_cs @ P_drift
                mu_c = mu_c @ P_cs @ P_drift
            total_phi += phi_c
            total_q[:, c] = mu_c * self._class_counts[c]

        reg = self.gamma * torch.eye(self.buffer_size, device=self.device)
        W = torch.linalg.solve(total_phi + reg, total_q)
        if init_mode: 
            self.normal_fc.weight.data[:W.shape[1]] = W.t().to(self.normal_fc.weight.dtype)
        else: 
            self.weight.data = F.normalize(W, p=2, dim=0)

    def forward(self, x, new_forward=False):
        ref = next(self.backbone.parameters())
        with autocast('cuda', enabled=False):
            h = self.buffer(self.backbone(x.to(device=ref.device, dtype=ref.dtype), new_forward=new_forward))
        return {'logits': h.to(self.weight.dtype) @ self.weight}

    def forward_normal_fc(self, x, new_forward=False):
        ref = next(self.backbone.parameters())
        with autocast('cuda', enabled=False):
            h = self.buffer(self.backbone(x.to(device=ref.device, dtype=ref.dtype), new_forward=new_forward))
        return {"logits": self.normal_fc(h.to(self.normal_fc.weight.dtype))['logits']}

    # Giữ nguyên các hàm noise/gpm
    def update_noise(self):
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].update_noise()
    def unfreeze_noise(self):
        for j in range(len(self.backbone.noise_maker)): self.backbone.noise_maker[j].unfreeze_incremental()
    def init_unfreeze(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_task_0()
            for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters(): p.requires_grad = True
    def collect_projections(self, mode='threshold', val=0.95):
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].compute_projection_matrix(mode=mode, val=val)
    def apply_gpm_to_grads(self, scale=1.0):
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].apply_gradient_projection(scale=scale)