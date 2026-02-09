import copy
import gc
import torch
from torch import nn
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
import numpy as np
from sklearn.metrics import accuracy_score  
import os
import random

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
        self.device = args['device']
        
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        self.buffer = RandomBuffer(self.feature_dim, self.buffer_size, self.device)
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        
        # W hiện tại
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), **factory_kwargs))
        # W_ref (Snapshot)
        self.register_buffer("w_ref", torch.zeros((self.buffer_size, 0), **factory_kwargs))
        # R (Inverse Covariance) - Nằm trên GPU để tránh OOM RAM
        self.register_buffer("R", torch.eye(self.buffer_size, **factory_kwargs) / self.gamma)
        self.class_means = [] 
       
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        self.prev_known_class = 0 
        
    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.prev_known_class = self.known_class 
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
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None: nn.init.constant_(new_fc.bias, 0.)
        self.normal_fc = new_fc

    def forward(self, x, new_forward=False):
        """Forward không chuẩn hóa - Dùng đặc trưng thô trực tiếp"""
        if new_forward: hyper_features = self.backbone(x, new_forward=True)
        else: hyper_features = self.backbone(x)
        
        # [BỎ CHUẨN HÓA]: Dùng trực tiếp đặc trưng thô
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {'logits': logits}
    def update_noise(self):
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].update_noise()
    def after_task_magmax_merge(self):
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].after_task_training()
    def unfreeze_noise(self):
        for j in range(len(self.backbone.noise_maker)): self.backbone.noise_maker[j].unfreeze_incremental()
    def init_unfreeze(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_task_0()
            if hasattr(self.backbone.blocks[j], 'norm1'):
                for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            if hasattr(self.backbone.blocks[j], 'norm2'):
                for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
        if hasattr(self.backbone, 'norm'):
            for p in self.backbone.norm.parameters(): p.requires_grad = True

    def forward_fc(self, features):
        features = features.to(self.weight.dtype) 
        return features @ self.weight

    #fit sinh mẫu giả
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Hàm Fit tối ưu: 
        1. Không chuẩn hóa.
        2. Không tính lại Mean (Truy xuất từ Global Centroids).
        """
        self.backbone.eval()
        # --- 1. TRÍCH XUẤT ĐẶC TRƯNG THÔ ---
        f_real = self.backbone(X.to(self.device)).float().cpu()
        Y_cpu = Y.cpu().float()

        X_pseudo_list = []
        Y_pseudo_list = []
        
        # --- 2. SINH MẪU GIẢ (Dùng Centroid đã tính trước ở Min.py) ---
        if self.prev_known_class > 0:
            unique_new = torch.argmax(Y_cpu, dim=1).unique().tolist()
            
            # Lấy danh sách Centroids của các lớp mới để so sánh seed
            # (Chỉ dùng Normalize tạm thời để tính Cosine Similarity chọn Seed)
            new_means_mat = torch.stack([self.class_means[c] for c in unique_new])
            samples_per_old = max(1, min(int(f_real.shape[0] / self.prev_known_class), 500))

            for c_old in range(self.prev_known_class):
                if self.class_means[c_old] is not None:
                    mu_old = self.class_means[c_old]
                    # Tìm Seed dựa trên hướng (Cosine) - Đây là cách chọn seed chuẩn nhất
                    sims = torch.matmul(F.normalize(new_means_mat, p=2, dim=1), 
                                        F.normalize(mu_old.unsqueeze(0), p=2, dim=1).T)
                    c_seed = unique_new[torch.argmax(sims).item()]
                    
                    mu_seed = self.class_means[c_seed]
                    mask_seed = (torch.argmax(Y_cpu, dim=1) == c_seed)
                    seed_feats = f_real[mask_seed]
                    
                    if seed_feats.shape[0] > 0:
                        idx = torch.randint(0, seed_feats.shape[0], (samples_per_old,))
                        # TỊNH TIẾN THÔ (Không chuẩn hóa kết quả)
                        f_fake = (seed_feats[idx] - mu_seed) + mu_old
                        X_pseudo_list.append(f_fake)
                        Y_pseudo_list.append(torch.zeros(samples_per_old, Y_cpu.shape[1]).index_fill_(1, torch.tensor(c_old), 1.0))

        # --- 3. GỘP & CẬP NHẬT RLS ---
        X_total = torch.cat([f_real] + X_pseudo_list).to(self.device)
        Y_total = torch.cat([Y_cpu] + Y_pseudo_list).to(self.device)

        if Y_total.shape[1] > self.weight.shape[1]:
            diff = Y_total.shape[1] - self.weight.shape[1]
            self.weight = torch.cat((self.weight, torch.zeros((self.buffer_size, diff), device=self.device)), dim=1)

        H = self.buffer(X_total)
        RHt = H @ self.R.T 
        A = RHt @ H.T 
        A.diagonal().add_(1.0) 
        
        try: K = RHt.T @ torch.inverse(A + 1e-7 * torch.eye(A.shape[0], device=self.device))
        except: K = torch.linalg.solve(A, RHt).T

        self.R.sub_(K @ RHt) 
        self.weight.add_(K @ (Y_total - (H @ self.weight)))
    
    
    
    # --- HÀM MỚI: Weight Merging (Gọi 1 lần cuối task) ---
    def weight_merging(self, alpha=0.2):
        if self.cur_task > 0 and self.w_ref.shape[1] > 0:
            print(f"--> [Weight Merging] Blending with alpha={alpha}...")
            old_cols = self.prev_known_class
            
            W_new = self.weight[:, :old_cols]
            W_ref = self.w_ref[:, :old_cols].to(self.weight.device)
            
            # W_final = (1-alpha) * W_new + alpha * W_ref
            self.weight[:, :old_cols] = (1 - alpha) * W_new + alpha * W_ref

    def extract_feature(self, x): return self.backbone(x)
    
    def collect_projections(self, mode='threshold', val=0.95):
        print(f"--> [IncNet] GPM Collect...")
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].compute_projection_matrix(mode, val)

    def apply_gpm_to_grads(self, scale=1.0):
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].apply_gradient_projection(scale)


    def forward_normal_fc(self, x, new_forward=False):
        """Sửa đồng nhất: Bỏ chuẩn hóa ở forward train noise"""
        if new_forward: h = self.backbone(x, new_forward=True)
        else: h = self.backbone(x)
        h = self.buffer(h.to(self.buffer.weight.dtype))
        return {"logits": self.normal_fc(h.to(self.normal_fc.weight.dtype))['logits']}