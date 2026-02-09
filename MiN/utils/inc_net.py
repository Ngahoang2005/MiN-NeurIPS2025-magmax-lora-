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
        self.class_vars = []
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        self.prev_known_class = 0 
        
    def update_fc(self, nb_classes):
        # Lưu Snapshot
        # if self.cur_task >= 0:
        #     self.w_ref = self.weight.clone().detach()
            
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
        if new_forward: hyper_features = self.backbone(x, new_forward=True)
        else: hyper_features = self.backbone(x)
        hyper_features = F.normalize(hyper_features.float(), p=2, dim=1)
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


    # @torch.no_grad()
    # def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
    #     # 1. Feature Extraction
    #     with autocast('cuda', enabled=False):
    #         X = self.backbone(X).float() 
    #         X = self.buffer(X) 
    #         X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()
            
    #         # --- Expand Weight ---
    #         num_targets = Y.shape[1]
    #         if num_targets > self.weight.shape[1]:
    #             increment_size = num_targets - self.weight.shape[1]
    #             tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device)
    #             self.weight = torch.cat((self.weight, tail), dim=1)
    #             if self.w_ref.shape[1] > 0 and self.w_ref.shape[1] < num_targets:
    #                  ref_tail = torch.zeros((self.w_ref.shape[0], num_targets - self.w_ref.shape[1]), device=self.weight.device)
    #                  self.w_ref = torch.cat((self.w_ref, ref_tail), dim=1)
    #         elif num_targets < self.weight.shape[1]:
    #             increment_size = self.weight.shape[1] - num_targets
    #             tail = torch.zeros((Y.shape[0], increment_size), device=Y.device)
    #             Y = torch.cat((Y, tail), dim=1)

    #         # --- RLS Update với BATCH CHUNKING để tránh OOM ---
    #         # Thay vì tính cả batch lớn, ta chia nhỏ để giảm kích thước ma trận 'term'
    #         CHUNK_SIZE = 32 # Giảm xuống nếu vẫn bị OOM
    #         num_samples = X.shape[0]
            
    #         for start in range(0, num_samples, CHUNK_SIZE):
    #             end = min(start + CHUNK_SIZE, num_samples)
    #             x_chunk = X[start:end]
    #             y_chunk = Y[start:end]

    #             # Xử lý R trực tiếp trên GPU nhưng với chunk nhỏ
    #             term = torch.eye(x_chunk.shape[0], device=x_chunk.device) + x_chunk @ self.R @ x_chunk.T
    #             jitter = 1e-7 * torch.eye(term.shape[0], device=term.device)
                
    #             try: 
    #                 K = torch.linalg.solve(term + jitter, x_chunk @ self.R)
    #                 K = K.T
    #             except: 
    #                 K = self.R @ x_chunk.T @ torch.inverse(term + jitter)
                
    #             self.R.sub_(K @ x_chunk @ self.R) # Sử dụng in-place subtraction
    #             self.weight.add_(K @ (y_chunk - x_chunk @ self.weight)) # In-place addition

    #         del X, Y, x_chunk, y_chunk, K, term, jitter
    #         gc.collect()

    #fit sinh mẫu giả
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Phiên bản FeTrIL¹ bám sát cấu trúc của bạn + Chống OOM cho ViT.
        """
        self.backbone.eval()
        # 1. Feature Extraction (Chia nhỏ batch để tránh OOM bộ nhớ GPU)
        features_list = []
        EXTRACT_BATCH = 128
        for start in range(0, X.shape[0], EXTRACT_BATCH):
            end = min(start + EXTRACT_BATCH, X.shape[0])
            x_batch = X[start:end].to(self.device)
            f_batch = self.backbone(x_batch)
            # Normalize ngay sau backbone (Normalize 1)
            f_batch = F.normalize(f_batch.float(), p=2, dim=1)
            features_list.append(f_batch.cpu())
            
        X_real_norm = torch.cat(features_list) # [N_real, 768]
        Y_cpu_all = Y.cpu().float()
        del X, features_list

        # --- SINH MẪU GIẢ (FETRIL¹ LOGIC) ---
        new_class_samples_dict = {}
        new_class_means_dict = {}
        
        # Tính Mean cho các lớp mới trong Batch hiện tại
        unique_classes = torch.argmax(Y_cpu_all, dim=1).unique()
        for c in unique_classes:
            c = c.item()
            while len(self.class_means) <= c: self.class_means.append(None)
            mask = (torch.argmax(Y_cpu_all, dim=1) == c)
            feats = X_real_norm[mask]
            mean_c = feats.mean(dim=0)
            self.class_means[c] = mean_c.detach() # Lưu cho task sau
            new_class_samples_dict[c] = feats
            new_class_means_dict[c] = mean_c

        X_pseudo_list = []
        Y_pseudo_list = []
        
        # Nếu đang ở task increment (>0), thực hiện sinh mẫu cho các lớp cũ
        if self.prev_known_class > 0 and len(new_class_samples_dict) > 0:
            available_new = list(new_class_samples_dict.keys())
            new_means_matrix = torch.stack([new_class_means_dict[k] for k in available_new])
            
            # Cân bằng: Mỗi lớp cũ sinh ra số mẫu bằng trung bình mẫu lớp mới trong batch
            samples_per_old = max(1, min(int(X_real_norm.shape[0] / self.prev_known_class), 500))

            for c_old in range(self.prev_known_class):
                if self.class_means[c_old] is not None:
                    # Chọn lớp Seed giống nhất (Cosine)
                    mean_old_vec = self.class_means[c_old]
                    similarities = torch.matmul(new_means_matrix, mean_old_vec)
                    c_seed = available_new[torch.argmax(similarities).item()]
                    
                    # Dịch chuyển (Translation)
                    seed_feats = new_class_samples_dict[c_seed]
                    mean_seed = new_class_means_dict[c_seed]
                    
                    # Lấy mẫu ngẫu nhiên từ seed để dịch chuyển
                    idx = torch.randint(0, seed_feats.shape[0], (samples_per_old,))
                    X_fake = (seed_feats[idx] - mean_seed) + mean_old_vec
                    
                    # Normalize lại mẫu giả (Normalize 2)
                    X_fake = F.normalize(X_fake, p=2, dim=1)
                    
                    Y_fake = torch.zeros(samples_per_old, Y_cpu_all.shape[1])
                    Y_fake[:, c_old] = 1.0
                    X_pseudo_list.append(X_fake)
                    Y_pseudo_list.append(Y_fake)

        # Gộp Thật + Giả
        if len(X_pseudo_list) > 0:
            X_total = torch.cat([X_real_norm] + X_pseudo_list).to(self.device)
            Y_total = torch.cat([Y_cpu_all] + Y_pseudo_list).to(self.device)
        else:
            X_total = X_real_norm.to(self.device); Y_total = Y_cpu_all.to(self.device)
        # ---------------------------------------------------------------------
        # [SANITY CHECK] KIỂM TRA ĐỘ HOÀN HẢO CỦA NORMALIZE
        # ---------------------------------------------------------------------
        avg_norm = torch.norm(X_total, p=2, dim=1).mean().item()
        print(f"\n[SANITY CHECK] Task {self.cur_task} | Samples: {X_total.shape[0]} | Mean L2 Norm: {avg_norm:.6f}")
        if abs(avg_norm - 1.0) > 1e-3:
            print(">>> CẢNH BÁO: Normalize đang bị lệch, kiểm tra lại logic!")
        # ---------------------------------------------------------------------
        # --- DUAL-FORM RLS UPDATE ---
        if Y_total.shape[1] > self.weight.shape[1]:
            tail = torch.zeros((self.weight.shape[0], Y_total.shape[1] - self.weight.shape[1]), device=self.device)
            self.weight = torch.cat((self.weight, tail), dim=1)

        H = self.buffer(X_total) # Qua Random Project + ReLU
        zero_ratio = (H == 0).float().mean().item()
        print(f"Sparsity: {zero_ratio:.2f}")
        # Công thức Woodbury (Dual-Form) cho tốc độ siêu nhanh trên GPU
        RHt = H @ self.R.T 
        A = RHt @ H.T 
        A.diagonal().add_(1.0) # Cộng Identity

        try:
            B = torch.linalg.solve(A, torch.eye(A.shape[0], device=self.device))
        except:
            B = torch.inverse(A + 1e-7 * torch.eye(A.shape[0], device=self.device))

        K = RHt.T @ B 
        self.R.sub_(K @ RHt) 
        self.weight.add_(K @ (Y_total - (H @ self.weight)))

        del X_total, Y_total, H, A, B, K, RHt
        gc.collect()
        torch.cuda.empty_cache()
    
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
        if new_forward: h = self.backbone(x, new_forward=True)
        else: h = self.backbone(x)
        # Chuẩn hóa đồng nhất trước khi vào Buffer
        h = F.normalize(h.float(), p=2, dim=1)
        h = self.buffer(h.to(self.buffer.weight.dtype))
        h = h.to(self.normal_fc.weight.dtype)
        return {"logits": self.normal_fc(h)['logits']}