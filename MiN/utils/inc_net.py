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
        [FINAL VERSION: FeTrIL¹ + Cosine Similarity + Dual-Form RLS]
        - FeTrIL Strategy: Chọn lớp mới có Cosine Similarity cao nhất để làm seed.
        - RLS: Dual-Form (Nhanh).
        """
        # 1. Feature Extraction & Normalization (L2 Norm theo Paper)
        with torch.no_grad():
            if X.device != self.device: X = X.to(self.device)
            features_backbone = self.backbone(X)
            # [PAPER QUAN TRỌNG]: L2 Normalization trước khi xử lý 
            features_backbone = F.normalize(features_backbone, p=2, dim=1)

        # Đưa về CPU để xử lý Logic sinh mẫu
        X_real_backbone = features_backbone.cpu().float()
        Y_cpu_all = Y.cpu().float()
        del X, features_backbone, Y

        # -------------------------------------------------------
        # BƯỚC 1: TÍNH TOÁN CENTROID CHO CLASS MỚI
        # -------------------------------------------------------
        new_class_samples_dict = {}     # Lưu mẫu (để copy)
        new_class_means_dict = {}       # Lưu mean (để tính cosine)
        
        if self.training:
            if not hasattr(self, 'class_means'): self.class_means = []
            
            unique_classes = torch.argmax(Y_cpu_all, dim=1).unique()
            for c in unique_classes:
                c = c.item()
                # Expand list nếu cần
                while len(self.class_means) <= c: self.class_means.append(None)
                
                mask = (torch.argmax(Y_cpu_all, dim=1) == c)
                feats = X_real_backbone[mask]
                
                # Tính Mean
                mean_c = feats.mean(dim=0)
                # Normalize Mean (để tính Cosine cho chuẩn)
                mean_c_norm = F.normalize(mean_c.unsqueeze(0), p=2, dim=1).squeeze(0)
                
                # Lưu lại
                self.class_means[c] = mean_c.detach() # Lưu mean gốc để dịch chuyển
                new_class_samples_dict[c] = feats
                new_class_means_dict[c] = mean_c_norm # Lưu mean chuẩn hóa để so sánh

        # -------------------------------------------------------
        # BƯỚC 2: SINH MẪU GIẢ (FeTrIL¹ - COSINE SELECTION)
        # -------------------------------------------------------
        X_pseudo_list = []
        Y_pseudo_list = []
        
        if self.prev_known_class > 0 and len(new_class_samples_dict) > 0:
            available_new_classes = list(new_class_samples_dict.keys())
            
            # Chuẩn bị Tensor để tính Cosine nhanh (Matrix Multiplication)
            # Stack các mean mới thành ma trận [Số class mới, 768]
            new_means_matrix = torch.stack([new_class_means_dict[k] for k in available_new_classes])
            
            # Tính số lượng mẫu giả (Dynamic Balancing)
            total_real = X_real_backbone.shape[0]
            samples_per_old = int(total_real / self.prev_known_class)
            samples_per_old = max(1, min(samples_per_old, 500)) # Cap an toàn

            for c_old in range(self.prev_known_class):
                if c_old < len(self.class_means) and self.class_means[c_old] is not None:
                    
                    # --- [LOGIC MỚI: TÌM SEED XỊN NHẤT BẰNG COSINE] ---
                    # Lấy mean cũ
                    mean_old_vec = self.class_means[c_old]
                    mean_old_norm = F.normalize(mean_old_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
                    
                    # Tính Cosine Similarity: Old_Mean . New_Means_Matrix.T
                    # Kết quả: Vector [Số class mới] chứa độ giống nhau
                    similarities = torch.matmul(new_means_matrix, mean_old_norm)
                    
                    # Chọn index có similarity cao nhất
                    best_match_idx = torch.argmax(similarities).item()
                    c_seed = available_new_classes[best_match_idx]
                    
                    # Lấy mẫu của class đó làm seed
                    seed = new_class_samples_dict[c_seed]
                    # --------------------------------------------------

                    # Expand cho đủ số lượng
                    while seed.shape[0] < samples_per_old: 
                        seed = torch.cat((seed, seed), dim=0)
                    seed = seed[:samples_per_old]
                    
                    # FeTrIL Translation Formula:
                    # X_fake = (Seed - Mean_Seed) + Mean_Old
                    # Lưu ý: Dùng Mean gốc (chưa normalize) để dịch chuyển cho đúng biên độ
                    mean_seed_orig = seed.mean(0)
                    X_fake = (seed - mean_seed_orig) + mean_old_vec
                    
                    # Vì các vector gốc đã được normalize ở đầu hàm, 
                    # nên X_fake sinh ra cũng sẽ nằm trên mặt cầu đơn vị (xấp xỉ).
                    # Để chắc ăn, normalize lại phát nữa cho chuẩn L2
                    X_fake = F.normalize(X_fake, p=2, dim=1)

                    # Label
                    Y_fake = torch.zeros(samples_per_old, Y_cpu_all.shape[1])
                    Y_fake[:, c_old] = 1.0
                    
                    X_pseudo_list.append(X_fake)
                    Y_pseudo_list.append(Y_fake)

        # Gộp dữ liệu
        if len(X_pseudo_list) > 0:
            X_total = torch.cat([X_real_backbone] + X_pseudo_list, dim=0)
            Y_total = torch.cat([Y_cpu_all] + Y_pseudo_list, dim=0)
        else:
            X_total = X_real_backbone; Y_total = Y_cpu_all

        # -------------------------------------------------------
        # BƯỚC 3: RLS UPDATE (DUAL FORM)
        # -------------------------------------------------------
        # Đưa hết lên GPU
        if self.R.device != self.device: self.R = self.R.to(self.device)
        self.weight = self.weight.to(self.device)
        X_total = X_total.to(self.device)
        Y_total = Y_total.to(self.device)

        # Expand weight
        if Y_total.shape[1] > self.weight.shape[1]:
            tail = torch.zeros((self.weight.shape[0], Y_total.shape[1] - self.weight.shape[1]), device=self.device)
            self.weight = torch.cat((self.weight, tail), dim=1)

        H = self.buffer(X_total) # [N, 16384]
        
        # Woodbury Update (Nhanh)
        RHt = H @ self.R.T 
        A = RHt @ H.T 
        A.diagonal().add_(1.0)

        try:
            B = torch.linalg.solve(A, torch.eye(A.shape[0], device=self.device))
        except:
            B = torch.inverse(A + 1e-6 * torch.eye(A.shape[0], device=self.device))

        K = RHt.T @ B 
        
        self.R.sub_(K @ RHt) 
        Prediction_Error = Y_total - (H @ self.weight)
        self.weight.add_(K @ Prediction_Error)

        del X_real_backbone, X_total, Y_total, H, A, B, K, RHt, Prediction_Error
        gc.collect()
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