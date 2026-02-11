import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torch.nn import functional as F
import scipy.stats as stats
import timm
import random
# [ADDED] Import autocast để kiểm soát precision thủ công
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
        
        # [MODIFIED] Dùng float32 thay vì double (tiết kiệm 50% VRAM)
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)

        self.reset_parameters()

    # @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # [ADDED] Đảm bảo input cùng kiểu với weight (tránh lỗi FP16 vs FP32)
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
        self.task_prototypes = []

        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # --- DUAL CLASSIFIERS ---
        # 1. Universal (Giữ R liên tục)
        self.fc_uni = None
        self.register_buffer("R_uni", torch.eye(self.buffer_size, device=self.device, dtype=torch.float32) / self.gamma)

        # 2. Specific (Sẽ Reset R mỗi khi fit task mới)
        self.fc_spec = None
        self.register_buffer("R_spec", torch.eye(self.buffer_size, device=self.device, dtype=torch.float32) / self.gamma)

        # 3. Normal FC (Cho SGD trong run)
        self.normal_fc = None

        self.cur_task = -1
        self.known_class = 0
        self.task_class_indices = {} 

    def update_fc(self, nb_classes):
        self.cur_task += 1
        start_class = self.known_class
        self.known_class += nb_classes
        self.task_class_indices[self.cur_task] = list(range(start_class, self.known_class))

        self.fc_uni = self.generate_fc(self.buffer_size, self.known_class, self.fc_uni)
        self.fc_spec = self.generate_fc(self.buffer_size, self.known_class, self.fc_spec)
        self.update_normal_fc(self.known_class)

    def generate_fc(self, in_dim, out_dim, old_fc=None):
        new_fc = SimpleLinear(in_dim, out_dim, bias=True)
        if old_fc is not None:
            nb_output = old_fc.out_features
            weight = copy.deepcopy(old_fc.weight.data)
            bias = copy.deepcopy(old_fc.bias.data)
            new_fc.weight.data[:nb_output] = weight
            new_fc.bias.data[:nb_output] = bias
        return new_fc

    def update_normal_fc(self, nb_classes):
        if self.cur_task == 0:
            self.normal_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
        else:
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            if self.normal_fc is not None:
                nb_old = self.normal_fc.out_features
                new_fc.weight.data[:nb_old] = self.normal_fc.weight.data
                new_fc.bias.data[:nb_old] = self.normal_fc.bias.data
                nn.init.constant_(new_fc.weight.data[nb_old:], 0.)
                nn.init.constant_(new_fc.bias.data[nb_old:], 0.)
            self.normal_fc = new_fc

    def reset_R_spec(self):
        self.R_spec = torch.eye(self.buffer_size, device=self.device, dtype=torch.float32) / self.gamma

    # --- CORE LOGIC: BATCH RLS (Sử dụng solve để tối ưu tốc độ) ---
    def _fit_RLS_batch(self, X, Y, fc_layer, R_matrix):
        # 1. Tính toán K (Kalman Gain) dùng linalg.solve (Nhanh và ổn định hơn inverse)
        # Solve: (I + X R X^T) K = X R
        term = torch.eye(X.shape[0], device=X.device) + X @ R_matrix @ X.T
        # K_bridge = (I + X R X^T)^-1 @ X @ R
        # Dùng solve để tìm K_bridge trực tiếp
        K_bridge = torch.linalg.solve(term, X @ R_matrix)
        
        # 2. Update R_matrix
        R_matrix = R_matrix - (R_matrix @ X.T @ K_bridge)
        
        # 3. Update Weight
        # Error = Y - XW
        W_curr = fc_layer.weight.data.T
        error = Y - X @ W_curr
        W_new = W_curr + R_matrix @ X.T @ error
        
        fc_layer.weight.data = W_new.T
        return R_matrix

    # --- CÁC HÀM FIT CHIẾN THUẬT ---
    
    # 1. Fit dùng Feature đã cache (DÙNG CÁI NÀY TRONG MIN.PY ĐỂ NHANH)
    @torch.no_grad()
    def fit_spec_direct(self, X_feat, Y):
        task_idxs = self.task_class_indices[self.cur_task]
        Y_task = Y[:, task_idxs]
        
        # Tạo một bản sao layer giả chỉ chứa các cột của task hiện tại để fit
        tmp_fc = nn.Linear(self.buffer_size, len(task_idxs), bias=False).to(self.device)
        tmp_fc.weight.data = self.fc_spec.weight.data[task_idxs, :]
        
        self.R_spec = self._fit_RLS_batch(X_feat, Y_task, tmp_fc, self.R_spec)
        
        # Gán ngược lại vào ma trận chung
        self.fc_spec.weight.data[task_idxs, :] = tmp_fc.weight.data

    @torch.no_grad()
    def fit_uni_direct(self, X_feat, Y):
        self.R_uni = self._fit_RLS_batch(X_feat, Y, self.fc_uni, self.R_uni)

    # 2. Fit dùng Ảnh (Chậm hơn vì qua backbone)
    @torch.no_grad()
    def fit_spec(self, X, Y):
        with torch.cuda.amp.autocast(enabled=False):
            feat = self.buffer(self.backbone(X)).float()
            self.fit_spec_direct(feat, Y)

    @torch.no_grad()
    def fit_uni(self, X, Y):
        with torch.cuda.amp.autocast(enabled=False):
            feat = self.buffer(self.backbone(X)).float()
            self.fit_uni_direct(feat, Y)

    def forward_normal_fc(self, x, new_forward: bool = False):
        hyper_features = self.backbone(x)
        out = self.normal_fc(self.buffer(hyper_features))
        return out

    def forward(self, x, **kwargs):
        return self.forward_tuna_combined(x)

    # --- HYBRID INFERENCE (Giữ nguyên logic chuẩn trước đó) ---
    def forward_tuna_combined(self, x):
        was_training = self.training
        self.eval()
        
        batch_size = x.shape[0]
        num_tasks = len(self.backbone.noise_maker[0].mu)
        
        # 1. Universal
        self.set_noise_mode(-2)
        with torch.no_grad():
            feat_uni = self.buffer(self.backbone(x))
            logits_uni = self.fc_uni(feat_uni)['logits'] 

        # 2. Specific & Routing
        min_entropy = torch.full((batch_size,), float('inf'), device=x.device)
        best_task_ids = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        saved_task_logits = [] 

        with torch.no_grad():
            for t in range(num_tasks):
                self.set_noise_mode(t)
                feat_t = self.buffer(self.backbone(x))
                l_t = self.fc_spec(feat_t)['logits'] 
                saved_task_logits.append(l_t)
                
                # Masked Entropy
                if t in self.task_class_indices:
                    task_cols = self.task_class_indices[t]
                    l_t_masked = l_t[:, task_cols] 
                    
                    # --- THÊM TEMPERATURE SCALING Ở ĐÂY ---
                    # scale_factor = 5.0 tương đương với Temperature T = 0.2
                    # Nó giúp phóng đại sự khác biệt nhỏ giữa các logits
                    scale_factor = 5.0 
                    prob = torch.softmax(l_t_masked * scale_factor, dim=1)
                    
                    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
                    mask = entropy < min_entropy
                    min_entropy[mask] = entropy[mask]
                    best_task_ids[mask] = t
        with torch.no_grad():
            # Tập hợp logits của các expert được chọn cho từng ảnh
            # Shape: [Batch, Total_Classes]
            best_spec_logits = torch.stack(saved_task_logits, dim=1)[range(batch_size), best_task_ids]
            
            # Tính Magnitude cho TỪNG ẢNH (dim=1)
            # keepdim=True để có shape [Batch, 1], giúp nhân broadcast dễ dàng
            mag_uni_sample = logits_uni.abs().mean(dim=1, keepdim=True)
            mag_spec_sample = best_spec_logits.abs().mean(dim=1, keepdim=True)
            
            # Alpha riêng cho mỗi mẫu: Alpha[i] = Mag_Uni[i] / Mag_Spec[i]
            alpha_sample = mag_uni_sample / (mag_spec_sample + 1e-8)
            
            # Giới hạn Alpha trong khoảng [0.1, 1.0] để bảo vệ nhánh Uni
            alpha_sample = torch.clamp(alpha_sample, min=0.2, max=1.0)

            if random.random() <1:
                # In ra trung bình batch để bạn vẫn theo dõi được xu hướng chung
                print(f">>> Avg Dynamic Alpha: {alpha_sample.mean().item():.4f}")

        # 3. Final Ensemble với Alpha riêng cho từng mẫu
        final_logits = logits_uni.clone()
        for t in range(num_tasks):
            if t in self.task_class_indices:
                class_idxs = self.task_class_indices[t]
                mask_t = (best_task_ids == t)
                if mask_t.sum() > 0:
                    # Lấy expert logits và alpha của các mẫu thuộc task t
                    expert_l = saved_task_logits[t][mask_t] # [num_masked, Total_Classes]
                    a_t = alpha_sample[mask_t]             # [num_masked, 1]
                    
                    # Chỉ cộng vào các cột của task này, có nhân scale riêng cho từng ảnh
                    # PyTorch tự broadcast a_t vào các cột của expert_l
                    final_logits[mask_t, class_idxs[0]:class_idxs[-1]+1] += a_t * expert_l[:, class_idxs]

        self.set_noise_mode(-2)
        if was_training: self.train()
        return {'logits': final_logits}
    def set_noise_mode(self, mode):
        if hasattr(self.backbone, 'noise_maker'):
            for m in self.backbone.noise_maker:
                m.active_task_idx = mode

    def extract_feature(self, x):
        return self.buffer(self.backbone(x))

    def update_task_prototype(self, prototype):
        if isinstance(prototype, torch.Tensor): self.task_prototypes[-1] = prototype.detach().cpu()
        else: self.task_prototypes[-1] = prototype

    def extend_task_prototype(self, prototype):
        if isinstance(prototype, torch.Tensor): self.task_prototypes.append(prototype.detach().cpu())
        else: self.task_prototypes.append(prototype)

    def update_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()
            self.backbone.noise_maker[j].init_weight_noise(self.task_prototypes)

    def unfreeze_noise(self):
    # Duyệt qua từng lớp Noise Maker trong Backbone
        for m in self.backbone.noise_maker:
            # 1. Lấy Expert hiện tại trong ModuleList
            expert_layer = m.mu[self.cur_task] 
            
            # 2. KÍCH HOẠT GRADIENT
            for param in expert_layer.parameters():
                param.requires_grad = True
                
            # 3. KHỞI TẠO NÓNG (Crucial Step)
            # Khởi tạo trọng số với một lượng nhiễu cực nhỏ (Xavier hoặc Normal)
            # Điều này giúp SGD có "đà" để tối ưu ngay từ batch đầu tiên
            torch.nn.init.normal_(expert_layer.weight, std=0.001) 
            if expert_layer.bias is not None:
                torch.nn.init.constant_(expert_layer.bias, 0.0)
            
        print(f">>> [System] Expert {self.cur_task} unfrozen and initialized with small noise.")
    def init_unfreeze(self):
        for j in range(self.backbone.layer_num):
            for param in self.backbone.noise_maker[j].parameters(): param.requires_grad = True
            for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
        for p in self.backbone.norm.parameters(): p.requires_grad = True






