import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast 

class BaseIncNet(nn.Module):
    def __init__(self, args: dict):
        super(BaseIncNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.feature_dim = self.backbone.out_dim
        self.fc = None

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    @staticmethod
    def generate_fc(in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def forward(self, x):
        hyper_features = self.backbone(x)
        logits = self.fc(hyper_features)['logits']
        return {'features': hyper_features, 'logits': logits}

class RandomBuffer(torch.nn.Linear):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features, self.out_features = in_features, buffer_size
        factory_kwargs = {"device": device, "dtype": torch.float32}
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)
        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # [FIX TYPE]: Ép về dtype của weight để nhân ma trận nhanh/chuẩn
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.weight)

class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args, self.device = args, args['device']
        self.backbone = get_pretrained_backbone(args).to(self.device)
        self.gamma, self.buffer_size = args['gamma'], args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)
        
        # [DPCR STORAGE]: Thay R bằng Dict để bất tử với Index/OOM
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), **factory_kwargs))
        
        self._compressed_stats = {} 
        self._mu_list = {}          
        self._class_counts = {}     
        self.temp_phi, self.temp_mu, self.temp_count = {}, {}, {}

        self.normal_fc = None
        self.cur_task, self.known_class = -1, 0

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        if self.cur_task > 0:
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
        if self.normal_fc is not None:
            old_nb_output = self.normal_fc.out_features
            new_fc.weight.data[:old_nb_output] = self.normal_fc.weight.data
            if new_fc.bias is not None and self.normal_fc.bias is not None:
                new_fc.bias.data[:old_nb_output] = self.normal_fc.bias.data
        
        self.normal_fc = new_fc.to(self.device)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Thay thế RLS bằng tích lũy thống kê DPCR, giữ nguyên signature Fit(X, Y)"""
        self.eval()
        # [FIX TYPE]: Đồng bộ hóa trước khi vào backbone
        ref_dtype = next(self.backbone.parameters()).dtype
        X = X.to(device=self.device, dtype=ref_dtype)
        
        # Tính toán đặc trưng (Backbone chạy Mixed Precision nếu đang trong autocast)
        feat = self.buffer(self.backbone(X))
        Y = Y.to(device=self.device, dtype=self.weight.dtype)

        labels = torch.argmax(Y, dim=1)
        for i in range(feat.shape[0]):
            label = labels[i].item()
            f = feat[i:i+1].float() # DPCR cần độ chính xác cao FP32
            if label not in self.temp_phi:
                self.temp_phi[label] = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
                self.temp_mu[label] = torch.zeros(self.buffer_size, device=self.device)
                self.temp_count[label] = 0
            self.temp_phi[label] += (f.t() @ f).detach()
            self.temp_mu[label] += f.squeeze(0).detach()
            self.temp_count[label] += 1

    @torch.no_grad()
    def compress_stats(self):
        """Hàm nén để tiết kiệm VRAM"""
        RANK = 256
        for label in sorted(list(self.temp_phi.keys())):
            S, V = torch.linalg.eigh(self.temp_phi[label])
            S_top, V_top = (S[-RANK:], V[:, -RANK:]) if S.shape[0] > RANK else (S, V)
            self._compressed_stats[label] = (V_top.cpu(), S_top.cpu())
            self._mu_list[label] = (self.temp_mu[label] / self.temp_count[label]).cpu()
            self._class_counts[label] = self.temp_count[label]
            del self.temp_phi[label], self.temp_mu[label]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def solve_analytic(self, P_drift=None, boundary=0, init_mode=False):
        """Bước giải hệ phương trình thay cho RLS"""
        all_keys = list(self._compressed_stats.keys()) + list(self.temp_phi.keys())
        num_total = max(all_keys) + 1 if all_keys else 0
        if num_total > self.weight.shape[1]:
            new_w = torch.zeros((self.buffer_size, num_total), device=self.device)
            new_w[:, :self.weight.shape[1]] = self.weight
            self.register_buffer("weight", new_w)

        total_phi = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
        total_q = torch.zeros(self.buffer_size, num_total, device=self.device)

        for c, (V, S) in self._compressed_stats.items():
            V, S = V.to(self.device), S.to(self.device)
            phi_c = (V @ torch.diag(S)) @ V.t()
            mu_c = self._mu_list[c].to(self.device)
            if P_drift is not None and c < boundary:
                P_drift = P_drift.to(self.device)
                P_cs = V @ V.t()
                phi_c = P_drift.t() @ P_cs.t() @ phi_c @ P_cs @ P_drift
                mu_c = mu_c @ P_cs @ P_drift
            total_phi += phi_c
            total_q[:, c] = mu_c * self._class_counts[c]
            del V, S, phi_c, mu_c

        for label, phi in self.temp_phi.items():
            total_phi += phi; total_q[:, label] = self.temp_mu[label]

        W = torch.linalg.solve(total_phi + self.gamma * torch.eye(self.buffer_size, device=self.device), total_q)
        if init_mode: self.normal_fc.weight.data[:W.shape[1]] = W.t().to(self.normal_fc.weight.dtype)
        else: self.weight.data = F.normalize(W, p=2, dim=0)

    def forward(self, x, new_forward=False):
        ref_dtype = next(self.backbone.parameters()).dtype
        h = self.buffer(self.backbone(x.to(dtype=ref_dtype), new_forward=new_forward))
        return {'logits': self.forward_fc(h)}

    def forward_fc(self, features):
        return features.to(self.weight.dtype) @ self.weight

    def forward_normal_fc(self, x, new_forward=False):
        ref_dtype = next(self.backbone.parameters()).dtype
        h = self.buffer(self.backbone(x.to(dtype=ref_dtype), new_forward=new_forward))
        return {"logits": self.normal_fc(h.to(self.normal_fc.weight.dtype))['logits']}

    # Giữ nguyên các hàm noise/gpm gốc
    def update_noise(self):
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].update_noise()
    def unfreeze_noise(self):
        for j in range(len(self.backbone.noise_maker)): self.backbone.noise_maker[j].unfreeze_incremental()
    def init_unfreeze(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_task_0()
            if hasattr(self.backbone.blocks[j], 'norm1'):
                for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            if hasattr(self.backbone.blocks[j], 'norm2'):
                for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters(): p.requires_grad = True
    def collect_projections(self, mode='threshold', val=0.95):
        print(f"--> [IncNet] Collecting Projections (Mode: {mode}, Val: {val})...")
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].compute_projection_matrix(mode=mode, val=val)
    def apply_gpm_to_grads(self, scale=1.0):
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].apply_gradient_projection(scale=scale)
    def extract_feature(self, x): return self.backbone(x.to(self.device))