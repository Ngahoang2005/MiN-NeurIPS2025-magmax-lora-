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

# class MiNbaseNet(nn.Module):
#     def __init__(self, args: dict):
#         super(MiNbaseNet, self).__init__()
#         self.args = args
#         self.backbone = get_pretrained_backbone(args)
#         self.device = args['device']
        
#         self.gamma = args['gamma']
#         self.buffer_size = args['buffer_size']
#         self.feature_dim = self.backbone.out_dim 

#         self.buffer = RandomBuffer(self.feature_dim, self.buffer_size, self.device)
#         factory_kwargs = {"device": self.device, "dtype": torch.float32}
        
#         # W hiện tại
#         self.register_buffer("weight", torch.zeros((self.buffer_size, 0), **factory_kwargs))
#         # W_ref (Snapshot)
#         self.register_buffer("w_ref", torch.zeros((self.buffer_size, 0), **factory_kwargs))
#         # R (Inverse Covariance) - Nằm trên GPU để tránh OOM RAM
#         self.register_buffer("R", torch.eye(self.buffer_size, **factory_kwargs) / self.gamma)
#         self.class_means = [] 
       
#         self.normal_fc = None
#         self.cur_task = -1
#         self.known_class = 0
#         self.prev_known_class = 0 
        
#     def update_fc(self, nb_classes):
#         self.cur_task += 1
#         self.prev_known_class = self.known_class 
#         self.known_class += nb_classes
        
#         if self.cur_task > 0:
#             new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
#         else:
#             new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
#         if self.normal_fc is not None:
#             old_nb_output = self.normal_fc.out_features
#             with torch.no_grad():
#                 new_fc.weight[:old_nb_output] = self.normal_fc.weight.data
#                 nn.init.constant_(new_fc.weight[old_nb_output:], 0.)
#             del self.normal_fc
#         else:
#             nn.init.constant_(new_fc.weight, 0.)
#             if new_fc.bias is not None: nn.init.constant_(new_fc.bias, 0.)
#         self.normal_fc = new_fc

#     def forward(self, x, new_forward=False):
#         """Forward không chuẩn hóa - Dùng đặc trưng thô trực tiếp"""
#         if new_forward: hyper_features = self.backbone(x, new_forward=True)
#         else: hyper_features = self.backbone(x)
        
#         # [BỎ CHUẨN HÓA]: Dùng trực tiếp đặc trưng thô
#         hyper_features = hyper_features.to(self.weight.dtype)
#         logits = self.forward_fc(self.buffer(hyper_features))
#         return {'logits': logits}
#     def update_noise(self):
#         for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].update_noise()
#     def after_task_magmax_merge(self):
#         for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].after_task_training()
#     def unfreeze_noise(self):
#         for j in range(len(self.backbone.noise_maker)): self.backbone.noise_maker[j].unfreeze_incremental()
#     def init_unfreeze(self):
#         for j in range(len(self.backbone.noise_maker)):
#             self.backbone.noise_maker[j].unfreeze_task_0()
#             if hasattr(self.backbone.blocks[j], 'norm1'):
#                 for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
#             if hasattr(self.backbone.blocks[j], 'norm2'):
#                 for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
#         if hasattr(self.backbone, 'norm'):
#             for p in self.backbone.norm.parameters(): p.requires_grad = True

#     def forward_fc(self, features):
#         features = features.to(self.weight.dtype) 
#         return features @ self.weight

#     #fit sinh mẫu giả
#     @torch.no_grad()
#     def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
#         self.backbone.eval()
#         # 1. Trích xuất đặc trưng THÔ
#         f_real = self.backbone(X.to(self.device)).float().cpu()
#         Y_cpu = Y.cpu().float()
        
#         # Lấy các lớp có trong batch hiện tại (thường là các lớp của task mới)
#         unique_new = torch.argmax(Y_cpu, dim=1).unique().tolist()

#         X_pseudo_list = []
#         Y_pseudo_list = []
        
#         # 2. Sinh mẫu giả (Chỉ tiêu thụ Centroid, không tính toán)
#         if self.prev_known_class > 0:
#             # Lấy danh sách means của các lớp mới để làm seed
#             # Đã đảm bảo self.class_means được lấp đầy từ Min.py
#             new_means_mat = torch.stack([self.class_means[c] for c in unique_new])
#             samples_per_old = max(1, min(int(f_real.shape[0] / self.prev_known_class), 500))

#             for c_old in range(self.prev_known_class):
#                 # mu_old PHẢI TỒN TẠI từ các task trước
#                 mu_old = self.class_means[c_old]
                
#                 # Chọn Seed (Cosine Similarity)
#                 sims = torch.matmul(F.normalize(new_means_mat, p=2, dim=1), 
#                                     F.normalize(mu_old.unsqueeze(0), p=2, dim=1).T)
#                 c_seed = unique_new[torch.argmax(sims).item()]
                
#                 mu_seed = self.class_means[c_seed]
#                 mask_seed = (torch.argmax(Y_cpu, dim=1) == c_seed)
#                 seed_feats = f_real[mask_seed]
                
#                 if seed_feats.shape[0] > 0:
#                     idx = torch.randint(0, seed_feats.shape[0], (samples_per_old,))
#                     # Tịnh tiến Euclid: $X_{fake} = (X_{seed} - \mu_{seed}) + \mu_{old}$
#                     f_fake = (seed_feats[idx] - mu_seed) + mu_old
#                     X_pseudo_list.append(f_fake)
#                     Y_pseudo_list.append(torch.zeros(samples_per_old, Y_cpu.shape[1]).index_fill_(1, torch.tensor(c_old), 1.0))

#         # 3. Gộp và RLS Update (Dữ liệu THÔ)
#         X_total = torch.cat([f_real] + X_pseudo_list).to(self.device)
#         Y_total = torch.cat([Y_cpu] + Y_pseudo_list).to(self.device)

#         # Cập nhật RLS (Giữ nguyên logic của bạn)
#         if Y_total.shape[1] > self.weight.shape[1]:
#             self.weight = torch.cat((self.weight, torch.zeros((self.buffer_size, Y_total.shape[1] - self.weight.shape[1]), device=self.device)), dim=1)

#         H = self.buffer(X_total)
#         RHt = H @ self.R.T 
#         A = RHt @ H.T 
#         A.diagonal().add_(1e-6) 
        
#         K = RHt.T @ torch.linalg.solve(A + torch.eye(A.shape[0], device=self.device), torch.eye(A.shape[0], device=self.device))
#         self.R.sub_(K @ H @ self.R) 
#         self.weight.add_(K @ (Y_total - (H @ self.weight)))
#     # --- HÀM MỚI: Weight Merging (Gọi 1 lần cuối task) ---
#     def weight_merging(self, alpha=0.2):
#         if self.cur_task > 0 and self.w_ref.shape[1] > 0:
#             print(f"--> [Weight Merging] Blending with alpha={alpha}...")
#             old_cols = self.prev_known_class
            
#             W_new = self.weight[:, :old_cols]
#             W_ref = self.w_ref[:, :old_cols].to(self.weight.device)
            
#             # W_final = (1-alpha) * W_new + alpha * W_ref
#             self.weight[:, :old_cols] = (1 - alpha) * W_new + alpha * W_ref

#     def extract_feature(self, x): return self.backbone(x)
    
#     def collect_projections(self, mode='threshold', val=0.95):
#         print(f"--> [IncNet] GPM Collect...")
#         for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].compute_projection_matrix(mode, val)

#     def apply_gpm_to_grads(self, scale=1.0):
#         for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].apply_gradient_projection(scale)


#     def forward_normal_fc(self, x, new_forward=False):
#         """Sửa đồng nhất: Bỏ chuẩn hóa ở forward train noise"""
#         if new_forward: h = self.backbone(x, new_forward=True)
#         else: h = self.backbone(x)
#         h = self.buffer(h.to(self.buffer.weight.dtype))
#         return {"logits": self.normal_fc(h.to(self.normal_fc.weight.dtype))['logits']}
    
# end GPM
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

    #fit sinh mẫu giả
    @torch.no_grad()
    def generate_pseudo_dataset_hybrid(self, train_loader):
        print("--> [FeTrIL] Generating Pseudo Data (Global Hybrid Mode)...")
        self.eval()
        
        all_features = []
        all_labels = []
        
        # --- [FIX LỖI UNPACK] ---
        # Loader trả về: (index, image, label) -> Dùng (_, inputs, labels)
        for _, inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            features = self.backbone(inputs).float() 
            all_features.append(features)
            all_labels.append(labels.to(self.device))
            
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)

        # ... (Phần còn lại giữ nguyên) ...
        
        # Logic tính Mean và Fallback Gaussian giữ nguyên như cũ
        unique_new = torch.unique(all_labels).tolist()
        self.new_class_means = [] 
        
        for c in unique_new:
            mask = (all_labels == c)
            mean_c = all_features[mask].mean(dim=0)
            self.class_means[c] = mean_c 

        unique_new.sort()
        new_means_mat = torch.stack([self.class_means[c] for c in unique_new])
        task_std = torch.std(all_features, dim=0).mean().item()
        
        pseudo_features = []
        pseudo_labels = []
        
        if self.prev_known_class > 0:
            THRESHOLD = 0.1
            samples_per_old = 200 
            
            for c_old in range(self.prev_known_class):
                mu_old = self.class_means[c_old].to(self.device)
                
                sims = torch.matmul(
                    F.normalize(new_means_mat, p=2, dim=1), 
                    F.normalize(mu_old.unsqueeze(0), p=2, dim=1).T
                )
                best_sim_val, best_idx = torch.max(sims, dim=0)
                best_sim_val = best_sim_val.item()
                best_class_new = unique_new[best_idx.item()]
                
                f_fake = None
                
                if best_sim_val > THRESHOLD:
                    mask_seed = (all_labels == best_class_new)
                    seed_feats = all_features[mask_seed]
                    mu_seed = self.class_means[best_class_new].to(self.device)
                    
                    if seed_feats.shape[0] > 0:
                        idx = torch.randint(0, seed_feats.shape[0], (samples_per_old,))
                        f_fake = (seed_feats[idx] - mu_seed) + mu_old
                
                if f_fake is None:
                    shrinkage = 0.6
                    safe_std = task_std * shrinkage
                    noise = torch.randn(samples_per_old, self.feature_dim).to(self.device)
                    f_fake = mu_old + (noise * safe_std)
                
                pseudo_features.append(f_fake)
                pseudo_labels.append(torch.full((samples_per_old,), c_old, device=self.device))

        if len(pseudo_features) > 0:
            final_features = torch.cat([all_features, torch.cat(pseudo_features)])
            final_labels = torch.cat([all_labels, torch.cat(pseudo_labels)])
        else:
            final_features = all_features
            final_labels = all_labels
            
        return final_features, final_labels
    # =========================================================================
    # [3] CLASSIFIER TRAINING (RLS - Analytic)
    # =========================================================================

    def fit_classifier_global(self, train_loader):
        """
        Hàm này thay thế hàm fit() cũ. 
        Được gọi 1 lần duy nhất sau khi train backbone xong.
        """
        # 1. Sinh dữ liệu (Real + Hybrid Pseudo)
        X_total, Y_total_indices = self.generate_pseudo_dataset_hybrid(train_loader)
        
        # One-hot encode labels
        Y_total = torch.zeros(Y_total_indices.shape[0], self.known_class, device=self.device)
        Y_total.scatter_(1, Y_total_indices.unsqueeze(1).long(), 1.0)
        
        # 2. Mở rộng trọng số RLS nếu có class mới
        if self.known_class > self.weight.shape[1]:
            added_dim = self.known_class - self.weight.shape[1]
            self.weight = torch.cat((self.weight, torch.zeros((self.buffer_size, added_dim), device=self.device)), dim=1)

        # 3. Giải RLS (Analytic Solution)
        # Vì giải offline 1 lần nên ta có thể dùng batch lớn hoặc giải toàn cục
        # Để tránh OOM với tập dữ liệu lớn, ta vẫn chia batch để update ma trận R và W
        
        batch_size = 512 # Batch lớn cho nhanh vì chỉ tính ma trận
        N = X_total.shape[0]
        
        print(f"--> [RLS] Fitting Classifier on {N} samples (Real + Pseudo)...")
        
        # Reset R nếu muốn (Optional: Nếu muốn học lại từ đầu mỗi task để tránh lỗi tích lũy)
        # self.R = torch.eye(self.buffer_size, device=self.device) / self.gamma
        
        permutation = torch.randperm(N)
        
        for i in range(0, N, batch_size):
            indices = permutation[i : i + batch_size]
            batch_x = X_total[indices]
            batch_y = Y_total[indices]
            
            # Qua Random Buffer
            H = self.buffer(batch_x)
            
            # RLS Update Formula
            RHt = H @ self.R.T 
            A = RHt @ H.T 
            A.diagonal().add_(1e-6) # Stability
            
            K = RHt.T @ torch.linalg.solve(A + torch.eye(A.shape[0], device=self.device), torch.eye(A.shape[0], device=self.device))
            
            self.R.sub_(K @ H @ self.R) 
            self.weight.add_(K @ (batch_y - (H @ self.weight)))
    # --- HÀM MỚI: Weight Merging (Gọi 1 lần cuối task) ---
    
    def update_noise(self):
        """
        Gọi khi bắt đầu Task mới.
        Kích hoạt chế độ Sequential Initialization trong PiNoise.
        """
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    def after_task_magmax_merge(self):
        """
        Gọi sau khi kết thúc Task.
        Kích hoạt việc LƯU (Save) và TRỘN (Merge) tham số theo MagMax.
        """
        print(f"--> [IncNet] Task {self.cur_task}: Triggering Parameter-wise MagMax Merging...")
        for j in range(self.backbone.layer_num):
            # Hàm này nằm trong PiNoise
            self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        """Chỉ mở khóa gradient cho các module Noise (cho các task > 0)"""
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()

    def init_unfreeze(self):
        """
        Mở khóa gradient cho Task 0.
        Bao gồm Noise modules và các lớp Normalization của Backbone để ổn định hơn.
        """
        for j in range(self.backbone.layer_num):
            # Unfreeze Noise
            self.backbone.noise_maker[j].unfreeze_noise()
            
            # Unfreeze LayerNorms trong từng Block ViT
            for p in self.backbone.blocks[j].norm1.parameters():
                p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters():
                p.requires_grad = True
                
        # Unfreeze LayerNorm cuối cùng
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

    # =========================================================================
    # [ANALYTIC LEARNING (RLS) SECTION]
    # =========================================================================

    def forward_fc(self, features):
        """Forward qua Analytic Classifier"""
        # Đảm bảo features cùng kiểu với trọng số RLS (float32)
        features = features.to(self.weight) 
        return features @ self.weight

    def forward(self, x, new_forward: bool = False):
        """
        Dùng cho Inference/Testing.
        Chạy qua backbone (đã merge noise) -> Buffer -> Analytic Classifier.
        """
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # Ép kiểu về float32 cho Buffer và Classifier
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        
        return {
            'logits': logits
        }

    def extract_feature(self, x):
        """Chỉ trích xuất đặc trưng từ Backbone"""
        return self.backbone(x)

    def forward_normal_fc(self, x, new_forward: bool = False):
        """
        Dùng cho Training (Gradient Descent).
        Chạy qua backbone -> Buffer -> Normal FC (trainable).
        """
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = self.buffer(hyper_features)
        
        # Ép kiểu để khớp với Normal FC (có thể là FP16 nếu autocast bật bên ngoài)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        
        logits = self.normal_fc(hyper_features)['logits']
        
        return {
            "logits": logits}