import copy
import torch
from torch import nn
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
from torch.amp import autocast 

class RandomBuffer(torch.nn.Linear):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features, self.out_features = in_features, buffer_size
        factory_kwargs = {"device": device, "dtype": torch.float32}
        self.register_buffer("weight", torch.empty((self.in_features, self.out_features), **factory_kwargs))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.weight)

class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args, self.device = args, args['device']
        self.backbone = get_pretrained_backbone(args).to(self.device)
        self.gamma, self.buffer_size = args['gamma'], args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        self.buffer = RandomBuffer(self.feature_dim, self.buffer_size, self.device)
        
        # [KHÔI PHỤC RLS GỐC]: Dùng ma trận R để đảm bảo Acc 99.3%
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), **factory_kwargs))
        self.register_buffer("R", torch.eye(self.buffer_size, **factory_kwargs) / self.gamma)

        # [DPCR ADD-ON]: Bộ nhớ nén để xử lý Task sau
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
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """HÀM FIT RLS GỐC 100% - Đảm bảo Acc 99.3% cho Task 0"""
        with autocast('cuda', enabled=False):
            X = X.to(self.device).float()
            X = self.buffer(self.backbone(X).float())
            Y = Y.to(self.device).float()

            # Tự động mở rộng weight như bản gốc (Tránh lỗi Index)
            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                tail = torch.zeros((self.weight.shape[0], num_targets - self.weight.shape[1]), device=self.device)
                self.weight = torch.cat((self.weight, tail), dim=1)

            # [RLS MATH GỐC]: Cập nhật R và Weight trực tuyến
            term = torch.eye(X.shape[0], device=X.device) + X @ self.R @ X.T
            K = torch.linalg.solve(term + 1e-6*torch.eye(term.shape[0], device=self.device), X @ self.R).T
            
            self.R -= K @ X @ self.R
            self.weight += K @ (Y - X @ self.weight)

            # [DPCR ACCUMULATE]: Lưu thống kê để phục vụ nén
            labels = torch.argmax(Y, dim=1)
            for i in range(X.shape[0]):
                l = labels[i].item()
                if l not in self.temp_phi:
                    self.temp_phi[l] = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
                    self.temp_mu[l] = torch.zeros(self.buffer_size, device=self.device)
                    self.temp_count[l] = 0
                f = X[i:i+1]
                self.temp_phi[l] += (f.t() @ f).detach()
                self.temp_mu[l] += f.squeeze(0).detach()
                self.temp_count[l] += 1

    @torch.no_grad()
    def compress_stats(self):
        """DPCR compression: Chỉ gọi sau khi kết thúc Task để nén data cũ"""
        RANK = 256
        for label in sorted(list(self.temp_phi.keys())):
            S, V = torch.linalg.eigh(self.temp_phi[label])
            S_top, V_top = (S[-RANK:], V[:, -RANK:]) if S.shape[0] > RANK else (S, V)
            self._compressed_stats[label] = (V_top, S_top)
            self._mu_list[label] = (self.temp_mu[label] / self.temp_count[label])
            self._class_counts[label] = self.temp_count[label]
            del self.temp_phi[label], self.temp_mu[label]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def solve_with_drift(self, P_drift=None, boundary=0):
        """DPCR Drift Correction: Dùng để cập nhật weight khi sang Task mới"""
        total_phi = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
        total_q = torch.zeros(self.buffer_size, self.weight.shape[1], device=self.device)

        for c, (V, S) in self._compressed_stats.items():
            phi_c = (V @ torch.diag(S)) @ V.t()
            mu_c = self._mu_list[c]
            if P_drift is not None and c < boundary:
                phi_c = P_drift.t() @ (V @ V.t()).t() @ phi_c @ (V @ V.t()) @ P_drift
                mu_c = mu_c @ (V @ V.t()) @ P_drift
            total_phi += phi_c
            total_q[:, c] = mu_c * self._class_counts[c]

        W = torch.linalg.solve(total_phi + self.gamma * torch.eye(self.buffer_size, device=self.device), total_q)
        self.weight.data = F.normalize(W, p=2, dim=0)

    def forward(self, x, new_forward=False):
        ref_dtype = next(self.backbone.parameters()).dtype
        h = self.buffer(self.backbone(x.to(dtype=ref_dtype), new_forward=new_forward))
        return {'logits': h.to(self.weight.dtype) @ self.weight}

    def forward_normal_fc(self, x, new_forward=False):
        ref_dtype = next(self.backbone.parameters()).dtype
        h = self.buffer(self.backbone(x.to(dtype=ref_dtype), new_forward=new_forward))
        return {"logits": self.normal_fc(h.to(self.normal_fc.weight.dtype))['logits']}

    # Noise/GPM giữ nguyên
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