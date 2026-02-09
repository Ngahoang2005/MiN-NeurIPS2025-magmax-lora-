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
        [FIXED OOM] Batching Feature Extraction.
        Thay vì đưa cả cục X (5000 ảnh) vào backbone, ta chia nhỏ ra chạy từng chút một.
        """
        # -------------------------------------------------------
        # BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG (CHIA BATCH ĐỂ TRÁNH OOM)
        # -------------------------------------------------------
        self.backbone.eval()
        features_list = []
        
        # Batch size an toàn cho ViT/ResNet khi chỉ inference
        EXTRACT_BATCH = 128 
        total_samples = X.shape[0]
        
        # Loop qua từng batch nhỏ
        for start in range(0, total_samples, EXTRACT_BATCH):
            end = min(start + EXTRACT_BATCH, total_samples)
            
            # 1. Lấy batch nhỏ và đưa lên GPU
            x_batch = X[start:end].to(self.device)
            
            # 2. Qua Backbone (Chỉ tốn VRAM cho 128 ảnh)
            f_batch = self.backbone(x_batch)
            
            # [Quan trọng] Đưa về CPU ngay lập tức để giải phóng VRAM
            features_list.append(f_batch.cpu()) 

        # Gộp lại thành 1 cục to ở CPU (RAM thường chịu tốt 5000 vector)
        X_real_backbone = torch.cat(features_list).float()
        
        # Xử lý Y (nếu Y đang ở GPU thì đưa về CPU)
        Y_cpu_all = Y.cpu().float()
        
        # Dọn dẹp
        del X, features_list
        torch.cuda.empty_cache()

        # -------------------------------------------------------
        # BƯỚC 2: LOGIC SINH MẪU (FETRIL) - CHẠY TRÊN CPU
        # -------------------------------------------------------
        new_class_samples_dict = {}
        new_class_means_raw = {} 
        new_class_means_norm = {}
        
        if self.training:
            if not hasattr(self, 'class_means'): self.class_means = []
            
            unique_classes = torch.argmax(Y_cpu_all, dim=1).unique()
            for c in unique_classes:
                c = c.item()
                while len(self.class_means) <= c: self.class_means.append(None)
                
                mask = (torch.argmax(Y_cpu_all, dim=1) == c)
                feats = X_real_backbone[mask]
                
                mean_c = feats.mean(dim=0)
                
                self.class_means[c] = mean_c.detach()
                new_class_samples_dict[c] = feats
                new_class_means_raw[c] = mean_c
                new_class_means_norm[c] = F.normalize(mean_c.unsqueeze(0), p=2, dim=1).squeeze(0)

        X_pseudo_list = []
        Y_pseudo_list = []
        
        if self.prev_known_class > 0 and len(new_class_samples_dict) > 0:
            available_new_classes = list(new_class_samples_dict.keys())
            new_means_matrix_norm = torch.stack([new_class_means_norm[k] for k in available_new_classes])
            
            total_real = X_real_backbone.shape[0]
            samples_per_old = int(total_real / self.prev_known_class)
            samples_per_old = max(1, min(samples_per_old, 500))

            for c_old in range(self.prev_known_class):
                if c_old < len(self.class_means) and self.class_means[c_old] is not None:
                    
                    # Cosine Selection
                    mean_old_raw = self.class_means[c_old]
                    mean_old_norm = F.normalize(mean_old_raw.unsqueeze(0), p=2, dim=1).squeeze(0)
                    
                    similarities = torch.matmul(new_means_matrix_norm, mean_old_norm)
                    best_idx = torch.argmax(similarities).item()
                    c_seed = available_new_classes[best_idx]
                    
                    # Generation
                    seed = new_class_samples_dict[c_seed]
                    mean_seed_raw = new_class_means_raw[c_seed]
                    
                    while seed.shape[0] < samples_per_old: 
                        seed = torch.cat((seed, seed), dim=0)
                    seed = seed[:samples_per_old]
                    
                    X_fake = (seed - mean_seed_raw) + mean_old_raw
                    Y_fake = torch.zeros(samples_per_old, Y_cpu_all.shape[1])
                    Y_fake[:, c_old] = 1.0
                    
                    X_pseudo_list.append(X_fake)
                    Y_pseudo_list.append(Y_fake)

        # Gộp dữ liệu (Vẫn ở CPU)
        if len(X_pseudo_list) > 0:
            X_total = torch.cat([X_real_backbone] + X_pseudo_list, dim=0)
            Y_total = torch.cat([Y_cpu_all] + Y_pseudo_list, dim=0)
        else:
            X_total = X_real_backbone; Y_total = Y_cpu_all

        # -------------------------------------------------------
        # BƯỚC 3: RLS UPDATE (DUAL FORM) - CHẠY TRÊN GPU
        # -------------------------------------------------------
        # Chỉ đưa ma trận Feature (số) lên GPU -> Rất nhẹ
        if self.R.device != self.device: self.R = self.R.to(self.device)
        self.weight = self.weight.to(self.device)
        
        # Expand weight
        if Y_total.shape[1] > self.weight.shape[1]:
            tail = torch.zeros((self.weight.shape[0], Y_total.shape[1] - self.weight.shape[1]), device=self.device)
            self.weight = torch.cat((self.weight, tail), dim=1)

        # Chunking cho RLS (Phòng hờ ma trận 10k x 10k quá to)
        # Với Dual Form, ta cần tính H @ R.T (N x D) @ (D x D)
        # N=10000, D=768 (sau backbone) -> Nhẹ.
        # Nhưng H là output của Buffer (D=16384). N x D = 10000 x 16384 floats = 650MB. Vẫn ổn.
        
        X_total = X_total.to(self.device)
        Y_total = Y_total.to(self.device)

        H = self.buffer(X_total) # Project lên 16384 chiều
        
        # Woodbury Identity
        RHt = H @ self.R.T 
        A = RHt @ H.T 
        A.diagonal().add_(1.0) 

        try:
            B = torch.linalg.solve(A, torch.eye(A.shape[0], device=self.device))
        except:
            B = torch.inverse(A + 1e-6 * torch.eye(A.shape[0], device=self.device))

        K = RHt.T @ B 
        
        self.R.sub_(K @ RHt) 
        self.weight.add_(K @ (Y_total - (H @ self.weight)))

        # Dọn dẹp sạch sẽ
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
        h = self.buffer(h.to(self.buffer.weight.dtype))
        h = h.to(self.normal_fc.weight.dtype)
        return {"logits": self.normal_fc(h)['logits']}