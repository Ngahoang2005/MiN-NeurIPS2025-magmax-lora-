import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast 
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear

class RandomBuffer(torch.nn.Linear):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        factory_kwargs = {"device": device, "dtype": torch.float32}
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)
        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Ép về float32 để nhân ma trận chính xác
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.weight)

class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), device=self.device))

        # [FIX INDEX]: Dùng Dict để Class ID luôn chuẩn, không bị append nhầm
        self._compressed_stats = {} 
        self._mu_list = {}          
        self._class_counts = {}     

        self.temp_phi = {}
        self.temp_mu = {}
        self.temp_count = {}

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        # Giữ nguyên logic bias của bạn
        new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=(self.cur_task==0))
        if self.normal_fc is not None:
            with torch.no_grad():
                new_fc.weight[:self.normal_fc.out_features] = self.normal_fc.weight.data
                if new_fc.bias is not None and self.normal_fc.bias is not None:
                    new_fc.bias[:self.normal_fc.out_features] = self.normal_fc.bias.data
        self.normal_fc = new_fc.to(self.device)

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
        for j in range(self.backbone.layer_num): 
            self.backbone.noise_maker[j].compute_projection_matrix(mode=mode, val=val)

    def apply_gpm_to_grads(self, scale=1.0):
        for j in range(self.backbone.layer_num): 
            self.backbone.noise_maker[j].apply_gradient_projection(scale=scale)

    @torch.no_grad()
    def accumulate_stats(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.eval()
        # [FIX TYPE]: Ép về đúng dtype của backbone trước khi chạy
        X = X.to(self.backbone.patch_embed.proj.weight.dtype)
        feat = self.buffer(self.backbone(X).float())
        
        labels = torch.argmax(Y, dim=1)
        for i in range(feat.shape[0]):
            label = labels[i].item()
            f = feat[i:i+1]
            if label not in self.temp_phi:
                self.temp_phi[label] = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
                self.temp_mu[label] = torch.zeros(self.buffer_size, device=self.device)
                self.temp_count[label] = 0
            self.temp_phi[label] += (f.t() @ f).detach()
            self.temp_mu[label] += f.squeeze(0).detach()
            self.temp_count[label] += 1

    @torch.no_grad()
    def compress_stats(self):
        COMPRESS_RANK = 256
        torch.cuda.empty_cache()
        for label in sorted(list(self.temp_phi.keys())):
            raw_phi = self.temp_phi[label]
            try:
                # Thử tính trên GPU
                S, V = torch.linalg.eigh(raw_phi)
            except RuntimeError: 
                # [CPU FALLBACK]: Nếu 16k gây OOM, đẩy sang CPU giải
                print(f"!!! GPU OOM on Class {label}, Falling back to CPU for 16k matrix...")
                S, V = torch.linalg.eigh(raw_phi.cpu())
                S, V = S.to(self.device), V.to(self.device)

            S_top, V_top = (S[-COMPRESS_RANK:], V[:, -COMPRESS_RANK:]) if S.shape[0] > COMPRESS_RANK else (S, V)
            self._compressed_stats[label] = (V_top.cpu(), S_top.cpu())
            self._mu_list[label] = (self.temp_mu[label] / self.temp_count[label]).cpu()
            self._class_counts[label] = self.temp_count[label]
            del self.temp_phi[label], self.temp_mu[label], raw_phi
        self.temp_phi, self.temp_mu, self.temp_count = {}, {}, {}
        torch.cuda.empty_cache()

    @torch.no_grad()
    def fit(self, P_drift=None, known_classes_boundary=0, init_mode=False):
        torch.cuda.empty_cache()
        all_keys = list(self._compressed_stats.keys()) + list(self.temp_phi.keys())
        num_total_classes = max(all_keys) + 1 if all_keys else 0
        
        if num_total_classes > self.weight.shape[1]:
            new_w = torch.zeros((self.buffer_size, num_total_classes), device=self.device)
            new_w[:, :self.weight.shape[1]] = self.weight
            self.register_buffer("weight", new_w)

        total_phi = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
        total_q = torch.zeros(self.buffer_size, num_total_classes, device=self.device)

        for c in sorted(self._compressed_stats.keys()):
            V, S = self._compressed_stats[c]
            V, S = V.to(self.device), S.to(self.device)
            phi_c = (V @ torch.diag(S)) @ V.t()
            mu_c = self._mu_list[c].to(self.device)
            if P_drift is not None and c < known_classes_boundary:
                P_cs = V @ V.t()
                phi_c = P_drift.t() @ P_cs.t() @ phi_c @ P_cs @ P_drift # Drift Correction
                mu_c = mu_c @ P_cs @ P_drift
            total_phi += phi_c
            total_q[:, c] = mu_c * self._class_counts[c]
            del V, S, phi_c, mu_c

        for label, phi in self.temp_phi.items():
            total_phi += phi
            total_q[:, label] = self.temp_mu[label]

        reg = self.gamma * torch.eye(self.buffer_size, device=self.device)
        # 16k matrix solve
        W = torch.linalg.solve(total_phi + reg, total_q)
        if init_mode: 
            self.normal_fc.weight.data[:W.shape[1]] = W.t()
        else: 
            self.weight.data = F.normalize(W, p=2, dim=0)
        torch.cuda.empty_cache()

    def forward(self, x, new_forward=False):
        # [FIX TYPE]: Đồng bộ Half/Float
        x = x.to(self.backbone.patch_embed.proj.weight.dtype)
        h = self.buffer(self.backbone(x, new_forward=new_forward))
        return {'logits': h.to(self.weight.dtype) @ self.weight}

    def forward_normal_fc(self, x, new_forward=False):
        x = x.to(self.backbone.patch_embed.proj.weight.dtype)
        h = self.buffer(self.backbone(x, new_forward=new_forward))
        return {"logits": self.normal_fc(h.to(self.normal_fc.weight.dtype))['logits']}
    
    def extract_feature(self, x):
        x = x.to(self.backbone.patch_embed.proj.weight.dtype)
        return self.backbone(x)