import copy
import logging
import math
import numpy as np
import torch
import gc
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
        
        # W hiện tại (sẽ liên tục được cập nhật bởi RLS)
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), **factory_kwargs))
        
        # [THÊM MỚI] W_ref: Snapshot để làm điểm neo cho L2-SP
        self.register_buffer("w_ref", torch.zeros((self.buffer_size, 0), **factory_kwargs))
        
        # Đăng ký R là Buffer để nó nằm luôn trên GPU, không chuyển qua lại CPU nữa.
        # Kích thước ~1GB (nếu buffer 16k), GPU chịu tốt.
        self.register_buffer("R", torch.eye(self.buffer_size, **factory_kwargs) / self.gamma)
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        self.prev_known_class = 0 # Để biết đâu là cột cũ
        
    def update_fc(self, nb_classes):
        # [THÊM MỚI] Lưu Snapshot trước khi sang task mới
        if self.cur_task >= 0:
            self.w_ref = self.weight.clone().detach()
            
        self.cur_task += 1
        self.prev_known_class = self.known_class # Chốt số lượng class cũ
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
        
        # Không Normalize X (để giữ độ lớn cho RLS)
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
    # def forward_fc(self, features):
    #     features = features.to(self.weight.dtype)
        
    #     # --- [FIX ACC TỤT] Cosine Normalization ---
    #     # 1. Chuẩn hóa Input Feature (về độ dài 1)
    #     features_norm = F.normalize(features, p=2, dim=1)
        
    #     # 2. Chuẩn hóa Weight (về độ dài 1)
    #     # Weight shape: [In_dim, Out_dim] -> Normalize theo dim 0 (cột)
    #     weight_norm = F.normalize(self.weight, p=2, dim=0)
        
    #     # 3. Tính Logits (Cosine Similarity)
    #     # Nhân thêm một scalar (tau) để logits không quá bé (thường chọn 10-30)
    #     tau = 20.0 
    #     return tau * (features_norm @ weight_norm)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        try: from torch.amp import autocast
        except ImportError: from torch.cuda.amp import autocast
        
        # [FIX] Không load R từ CPU nữa, dùng trực tiếp self.R trên GPU
        
        with autocast('cuda', enabled=False):
            X = self.backbone(X).float() 
            X = self.buffer(X) 
            
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()
            
            # --- Expand Weight ---
            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device)
                self.weight = torch.cat((self.weight, tail), dim=1)
                
                # Expand w_ref nếu cần
                if self.w_ref.shape[1] > 0 and self.w_ref.shape[1] < num_targets:
                     ref_tail = torch.zeros((self.w_ref.shape[0], num_targets - self.w_ref.shape[1]), device=self.weight.device)
                     self.w_ref = torch.cat((self.w_ref, ref_tail), dim=1)

            elif num_targets < self.weight.shape[1]:
                # Pad Y với số 0
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size), device=Y.device)
                Y = torch.cat((Y, tail), dim=1)
            
            # --- RLS Update ---
            # Dùng self.R trực tiếp
            term = torch.eye(X.shape[0], device=X.device) + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            try: K = torch.linalg.solve(term + jitter, X @ self.R); K = K.T
            except: K = self.R @ X.T @ torch.inverse(term + jitter)
            
            # Update in-place
            self.R -= K @ X @ self.R
            self.weight += K @ (Y - X @ self.weight)
            
            # --- Soft L2-SP ---
            if self.cur_task > 0 and self.w_ref.shape[1] > 0:
                beta = 0.005 # Giữ mức thấp an toàn
                old_cols = self.prev_known_class
                current_old_W = self.weight[:, :old_cols]
                ref_old_W = self.w_ref[:, :old_cols].to(self.weight.device)
                self.weight[:, :old_cols] = (1 - beta) * current_old_W + beta * ref_old_W

            # [FIX] Không save R về CPU nữa
            
            del term, jitter, K, X, Y
            if torch.cuda.is_available(): torch.cuda.empty_cache()    
    def extract_feature(self, x): return self.backbone(x)
    
    def forward_normal_fc(self, x, new_forward=False):
        if new_forward: h = self.backbone(x, new_forward=True)
        else: h = self.backbone(x)
        h = self.buffer(h.to(self.buffer.weight.dtype))
        h = h.to(self.normal_fc.weight.dtype)
        return {"logits": self.normal_fc(h)['logits']}

    def collect_projections(self, mode='threshold', val=0.95):
        print(f"--> [IncNet] GPM Collect...")
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].compute_projection_matrix(mode, val)

    def apply_gpm_to_grads(self, scale=1.0):
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].apply_gradient_projection(scale)