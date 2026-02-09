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
        Phiên bản KHÔNG sửa RandomBuffer:
        - Giữ RandomBuffer trên GPU.
        - Đẩy từng batch dữ liệu nhỏ lên GPU để qua Buffer, sau đó kéo về CPU để tính RLS.
        """
        
        # --- BƯỚC 0: FEATURE EXTRACTION ---
        with torch.no_grad():
            try: from torch.cuda.amp import autocast
            except: pass
            
            if X.device != self.device: X = X.to(self.device)
            features_backbone = self.backbone(X) 
        
        X_real_backbone = features_backbone.cpu().float()
        Y_cpu_all = Y.cpu().float()
        
        del X, features_backbone, Y
        
        # --- BƯỚC 1: CẬP NHẬT THỐNG KÊ ---
        new_class_samples_dict = {}
        if self.training:
            unique_classes = torch.argmax(Y_cpu_all, dim=1).unique()
            for c in unique_classes:
                c = c.item()
                if c >= len(self.class_means):
                    while len(self.class_means) <= c:
                        self.class_means.append(None)
                        self.class_vars.append(None)
                
                mask = (torch.argmax(Y_cpu_all, dim=1) == c)
                features_c = X_real_backbone[mask]
                
                self.class_means[c] = features_c.mean(dim=0).detach()
                self.class_vars[c] = (features_c.var(dim=0, unbiased=False) + 1e-5).detach()
                new_class_samples_dict[c] = features_c

        # --- BƯỚC 2: SINH DATA GIẢ (FeTrIL) ---
        X_pseudo_list = []
        Y_pseudo_list = []
        
        if self.prev_known_class > 0 and len(new_class_samples_dict) > 0:
            available_new_classes = list(new_class_samples_dict.keys())
            
            num_new_samples_total = X_real_backbone.shape[0]
            num_new_classes_count = len(available_new_classes)
            
            if num_new_classes_count > 0:
                samples_per_old_class = int(num_new_samples_total / num_new_classes_count)
            else:
                samples_per_old_class = 1
            
            samples_per_old_class = max(samples_per_old_class, 1)

            for c_old in range(self.prev_known_class):
                if c_old < len(self.class_means) and self.class_means[c_old] is not None:
                    c_new_seed = np.random.choice(available_new_classes)
                    seed_features = new_class_samples_dict[c_new_seed]
                    
                    while seed_features.shape[0] < samples_per_old_class:
                        seed_features = torch.cat((seed_features, seed_features), dim=0)
                    seed_features = seed_features[:samples_per_old_class].clone()
                    
                    mean_new_source = seed_features.mean(dim=0)
                    mean_old_target = self.class_means[c_old]
                    
                    X_fake_backbone = (seed_features - mean_new_source) + mean_old_target
                    
                    Y_fake = torch.zeros(samples_per_old_class, Y_cpu_all.shape[1])
                    Y_fake[:, c_old] = 1.0
                    
                    X_pseudo_list.append(X_fake_backbone)
                    Y_pseudo_list.append(Y_fake)
        
        # --- BƯỚC 3: CHUẨN BỊ RLS ---
        if len(X_pseudo_list) > 0:
            X_pseudo_backbone = torch.cat(X_pseudo_list, dim=0)
            Y_pseudo = torch.cat(Y_pseudo_list, dim=0)
            X_total_backbone = torch.cat((X_real_backbone, X_pseudo_backbone), dim=0)
            Y_total = torch.cat((Y_cpu_all, Y_pseudo), dim=0)
        else:
            X_total_backbone = X_real_backbone
            Y_total = Y_cpu_all

        # [KHÁC BIỆT Ở ĐÂY] KHÔNG gọi self.buffer.cpu() nữa!
        # Để Buffer nằm yên trên GPU (self.device)
        
        BATCH_CHUNK = 4096
        total_samples = X_total_backbone.shape[0]
        
        # R và Weight vẫn ở CPU để tính toán cho nhẹ RAM
        if self.R.device.type != 'cpu': self.R = self.R.cpu()
        weight_cpu = self.weight.cpu()
        
        # Expand Weight
        num_targets = Y_total.shape[1]
        if num_targets > weight_cpu.shape[1]:
            tail = torch.zeros((weight_cpu.shape[0], num_targets - weight_cpu.shape[1]))
            weight_cpu = torch.cat((weight_cpu, tail), dim=1)
        elif num_targets < weight_cpu.shape[1]:
            tail = torch.zeros((Y_total.shape[0], weight_cpu.shape[1] - num_targets))
            Y_total = torch.cat((Y_total, tail), dim=1)

        # --- BƯỚC 4: RLS LOOP (GPU PROJECTION -> CPU MATH) ---
        for start_idx in range(0, total_samples, BATCH_CHUNK):
            end_idx = min(start_idx + BATCH_CHUNK, total_samples)
            
            # 1. Lấy batch nhỏ (đang ở CPU)
            X_batch_cpu = X_total_backbone[start_idx:end_idx]
            Y_batch = Y_total[start_idx:end_idx]
            
            # 2. Đẩy lên GPU để qua Buffer (tránh lỗi RandomBuffer)
            X_batch_gpu = X_batch_cpu.to(self.device)
            
            with torch.no_grad():
                # Buffer tính trên GPU (Nhanh & Không lỗi)
                X_batch_expanded_gpu = self.buffer(X_batch_gpu)
                
                # 3. Kéo kết quả về CPU ngay lập tức
                X_batch_expanded = X_batch_expanded_gpu.cpu()
            
            # 4. Tính RLS trên CPU (An toàn RAM)
            term = torch.eye(X_batch_expanded.shape[0]) + X_batch_expanded @ self.R @ X_batch_expanded.T
            jitter = 1e-6 * torch.eye(term.shape[0])
            
            try: 
                K = torch.linalg.solve(term + jitter, X_batch_expanded @ self.R)
                K = K.T
            except: 
                K = self.R @ X_batch_expanded.T @ torch.inverse(term + jitter)
            
            self.R -= K @ X_batch_expanded @ self.R
            weight_cpu += K @ (Y_batch - X_batch_expanded @ weight_cpu)
            
            # Xóa biến tạm
            del X_batch_gpu, X_batch_expanded_gpu, term, jitter, K, X_batch_cpu, X_batch_expanded, Y_batch

        # Đẩy Weight cuối cùng về lại GPU
        self.weight = weight_cpu.to(self.device)
        
        del X_real_backbone, X_total_backbone, Y_total, weight_cpu, new_class_samples_dict
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