import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
# Import autocast
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
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        hyper_features = self.backbone(x)
        logits = self.fc(hyper_features)['logits']
        return {
            'features': hyper_features,
            'logits': logits
        }


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
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.W)


class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        def forward_features_with_noise(self_bb, x, new_forward=False):
            # 1. Patch Embedding & Positional Embedding
            x = self_bb.patch_embed(x)
            x = self_bb._pos_embed(x)
            x = self_bb.patch_drop(x)
            x = self_bb.norm_pre(x)

            # 2. Loop qua từng Block + Noise
            for i in range(len(self_bb.blocks)):
                # Chạy qua Block Transformer gốc
                x = self_bb.blocks[i](x)
                
                # [QUAN TRỌNG] Chạy qua Noise (nếu có)
                if hasattr(self_bb, 'noise_maker'):
                    # Tương thích với logic của PiNoise (forward_new hoặc call thường)
                    if new_forward and hasattr(self_bb.noise_maker[i], 'forward_new'):
                        x = self_bb.noise_maker[i].forward_new(x)
                    else:
                        x = self_bb.noise_maker[i](x)
            
            # 3. Final Norm
            x = self_bb.norm(x)
            return x

        # Gán đè hàm của object (MethodType bind hàm vào instance)
        self.backbone.forward_features = type.MethodType(forward_features_with_noise, self.backbone)
        print("--> [SYSTEM] Backbone 'forward_features' has been MONKEY PATCHED to support Noise!")
        # =========================================================================------------
        self.device = args['device']
        
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        # Random Buffer
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # [RLS GỐC] Các tham số cho Analytic Learning cũ (dùng cho hàm fit)
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), **factory_kwargs))
        self.register_buffer("R", torch.eye(self.buffer_size, **factory_kwargs) / self.gamma)

        # --- [DPCR STORAGE] Lưu trữ thống kê Backbone (768-dim) ---
        self._saved_mean = {} 
        self._saved_cov = {}
        self._saved_count = {}

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        
        if self.cur_task > 0:
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
        if self.normal_fc is not None:
            old_nb_output = self.normal_fc.out_features
            with torch.no_grad():
                new_fc.weight[:old_nb_output] = self.normal_fc.weight.data
                nn.init.constant_(new_fc.weight[old_nb_output:], 0.)
            
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None:
                nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    def update_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    def unfreeze_noise(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_incremental()

    def init_unfreeze(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_task_0()
            if hasattr(self.backbone.blocks[j], 'norm1'):
                for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            if hasattr(self.backbone.blocks[j], 'norm2'):
                for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
                
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters(): p.requires_grad = True

    def forward_fc(self, features):
        features = features.to(self.weight.dtype) 
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        [RLS GỐC - KHÔNG SỬA]: Giữ nguyên để dùng cho bước Warm-up / Task 0.
        """
        with autocast('cuda', enabled=False):
            X_input = X.to(self.device).float()
            # Feature Extraction (Backbone -> Buffer)
            feat = self.backbone(X_input).float()
            feat = self.buffer(feat) 
            
            Y = Y.to(self.weight.device).float()

            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.buffer_size, increment_size), device=self.weight.device)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.weight.shape[1]:
                tail = torch.zeros((Y.shape[0], self.weight.shape[1] - num_targets), device=Y.device)
                Y = torch.cat((Y, tail), dim=1)

            # RLS Update (Recursive)
            term = torch.eye(feat.shape[0], device=feat.device) + feat @ self.R @ feat.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            try:
                K = torch.linalg.solve(term + jitter, feat @ self.R).T
            except:
                K = self.R @ feat.T @ torch.inverse(term + jitter)

            self.R -= K @ feat @ self.R
            self.weight += K @ (Y - feat @ self.weight)
            
            del term, jitter, K, X_input, Y, feat
            torch.cuda.empty_cache()

    # --- [DPCR IMPLEMENTATION: TSSP -> CIP -> RECONSTRUCTION] ---

    @torch.no_grad()
    def update_backbone_stats(self, X: torch.Tensor, Y: torch.Tensor):
        """Gom thống kê Mean/Cov (768-d) để dùng cho DPCR sau này."""
        self.eval()
        ref_p = next(self.backbone.parameters())
        X = X.to(device=self.device, dtype=ref_p.dtype)
        
        # Lấy feature BACKBONE (chưa qua Buffer)
        features = self.backbone(X).float() 
        
        if Y.dim() > 1: labels = torch.argmax(Y, dim=1)
        else: labels = Y
        
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            l = label.item()
            mask = (labels == l)
            f_sub = features[mask]
            
            if l not in self._saved_mean:
                self._saved_mean[l] = torch.zeros(self.feature_dim, device=self.device)
                self._saved_cov[l] = torch.zeros(self.feature_dim, self.feature_dim, device=self.device)
                self._saved_count[l] = 0
            
            self._saved_mean[l] += f_sub.sum(dim=0)
            self._saved_cov[l] += f_sub.t() @ f_sub
            self._saved_count[l] += f_sub.shape[0]

    @torch.no_grad()
    def solve_dpcr(self, P_drift=None, boundary=0):
        """
        Thực hiện quy trình DPCR: TSSP (Drift) -> CIP (Sampling) -> Replay Data.
        """
        if not self._saved_mean:
            return (torch.zeros(self.buffer_size, self.buffer_size, device=self.device),
                    torch.zeros(self.buffer_size, 0, device=self.device))

        num_total_known = max(list(self._saved_mean.keys())) + 1
        
        # Mở rộng classifier nếu cần (để khớp size)
        if num_total_known > self.weight.shape[1]:
            new_w = torch.zeros((self.buffer_size, num_total_known), device=self.device)
            new_w[:, :self.weight.shape[1]] = self.weight
            self.register_buffer("weight", new_w)

        HTH = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
        HTY = torch.zeros(self.buffer_size, num_total_known, device=self.device)

        if P_drift is not None: P_drift = P_drift.to(self.device)
        LAMBDA_SHRINK = 0.05 

        for c, count in self._saved_count.items():
            # 1. Load Stats cũ
            mean_vec = self._saved_mean[c] / count
            sec_mom = self._saved_cov[c] / count
            cov_mat = sec_mom - torch.outer(mean_vec, mean_vec)
            
            # Shrinkage (tăng ổn định cho CIP sampling)
            eye = torch.eye(cov_mat.shape[0], device=self.device)
            cov_mat = (1 - LAMBDA_SHRINK) * cov_mat + LAMBDA_SHRINK * eye

            # 2. [TSSP]: Áp dụng Global Drift P
            if P_drift is not None and c < boundary:
                mean_vec = mean_vec @ P_drift
                cov_mat = P_drift.t() @ cov_mat @ P_drift
            
            # 3. [CIP]: Sampling từ Covariance đã sửa
            # Đây là cách CIP hoạt động trên mạng phi tuyến: dùng Covariance định hình Subspace
            num_sample = min(count, 500) 
            try:
                dist = torch.distributions.MultivariateNormal(mean_vec, covariance_matrix=cov_mat)
                z = dist.sample((num_sample,))
            except:
                z = torch.randn(num_sample, mean_vec.shape[0], device=self.device) * 0.1 + mean_vec

            # 4. Projection qua Buffer (Non-linear)
            h_pseudo = self.buffer(z.float()) 
            
            # 5. Tích lũy cho Ridge Regression
            scale_factor = count / num_sample
            HTH += (h_pseudo.t() @ h_pseudo) * scale_factor
            
            Y_pseudo = torch.zeros(num_sample, num_total_known, device=self.device)
            Y_pseudo[:, c] = 1.0
            HTY += (h_pseudo.t() @ Y_pseudo) * scale_factor

        return HTH, HTY

    @torch.no_grad()
    def simple_ridge_solve(self, HTH, HTY):
        """Giải Ridge Regression (Eq. 19 trong bài báo)"""
        reg = self.gamma * torch.eye(self.buffer_size, device=self.device)
        W = torch.linalg.solve(HTH + reg, HTY)
        return W

    @torch.no_grad()
    def category_normalization(self, weight):
        """[CN] Category-wise Normalization (Eq. 22 trong bài báo)"""
        norms = torch.norm(weight, p=2, dim=0) # [Num_Classes]
        mean_norm = norms.mean()
        norms[norms == 0] = 1.0
        normalized_weight = weight * (mean_norm / norms)
        return normalized_weight
    def forward(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # [SỬA]: Đảm bảo đặc trưng đồng nhất kiểu dữ liệu trước khi vào Buffer
        hyper_features = hyper_features.to(self.weight.dtype)
        
        # Buffer trả về ReLU(X @ W), forward_fc thực hiện X @ Weight
        logits = self.forward_fc(self.buffer(hyper_features))
        
        return {'logits': logits}
    def extract_feature(self, x):
        """Chỉ trích xuất đặc trưng từ Backbone"""
        return self.backbone(x)
    # Forward Passes
    def forward_normal_fc(self, x, new_forward: bool = False):
        if new_forward: hyper_features = self.backbone(x, new_forward=True)
        else: hyper_features = self.backbone(x)
        
        hyper_features = self.buffer(hyper_features.to(self.buffer.weight.dtype))
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}

    def extract_feature(self, x):
        return self.backbone(x)

    def collect_projections(self, mode='threshold', val=0.95):
        print(f"--> [IncNet] Collecting Projections (Mode: {mode}, Val: {val})...")
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].compute_projection_matrix(mode=mode, val=val)

    def apply_gpm_to_grads(self, scale=1.0):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].apply_gradient_projection(scale=scale)