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
        # initiate params
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim  # dim of backbone
        self.task_prototypes = []

        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # [MODIFIED] Chuyển toàn bộ sang float32
        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

        self.Pinoise_list = nn.ModuleList()

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

        self.fc2 = nn.ModuleList()

    # [ADDED] Hỗ trợ Gradient Checkpointing (Cứu cánh cho OOM)
    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)
        elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'set_grad_checkpointing'):
             self.backbone.model.set_grad_checkpointing(enable)
        elif hasattr(self.backbone, 'grad_checkpointing'):
            self.backbone.grad_checkpointing = enable

    def forward_fc(self, features):
        features = features.to(self.weight)
        return features @ self.weight

    @property
    def in_features(self) -> int:
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        if self.cur_task > 0:
            fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
        if self.normal_fc is None:
            self.normal_fc = fc
        else:
            nn.init.constant_(fc.weight, 0.)

            del self.normal_fc
            self.normal_fc = fc

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        with autocast(enabled=False):
            # 1. Prepare Data (Giữ nguyên)
            X = self.backbone(X).float()
            X = self.buffer(X) 
            
            # Chuyển device và dtype một lần
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            # 2. Resize output nếu cần (Giữ nguyên logic)
            num_targets = Y.shape[1]
            if num_targets > self.out_features:
                increment_size = num_targets - self.out_features
                tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device, dtype=self.weight.dtype)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.out_features:
                increment_size = self.out_features - num_targets
                tail = torch.zeros((Y.shape[0], increment_size), device=Y.device, dtype=Y.dtype)
                Y = torch.cat((Y, tail), dim=1)

            # --- OPTIMIZATION STARTS HERE ---
            
            # [Tối ưu 1] Tính trước R * X.T (Ma trận Gain chưa chuẩn hóa)
            # Kích thước: (Feature_Dim, Batch_Size)
            # Thay vì tính X @ R nhiều lần, ta tính 1 lần và tái sử dụng.
            R_XT = self.R @ X.T

            # [Tối ưu 2] Tính term = I + X @ R @ X.T
            # X @ R @ X.T chính là X @ (R @ X.T) -> X @ R_XT
            # Kích thước: (Batch_Size, Batch_Size)
            term = X @ R_XT
            
            # [Tối ưu 3] Thêm Jitter và Identity bằng phép cộng tại chỗ (In-place)
            # Nhanh hơn nhiều so với việc tạo ma trận I và ma trận Jitter mới rồi cộng vào
            term.diagonal().add_(1.0 + 1e-6)

            # Nghịch đảo (Giữ nguyên logic inverse)
            K_inv = torch.inverse(term)

            # [Tối ưu 4] Tính Kalman Gain (K_gain)
            # K_gain = R_new @ X.T = (R_old @ X.T) @ inv(term)
            # Việc này giúp ta update weight và R mà không cần nhân lại ma trận lớn
            Gain = R_XT @ K_inv

            # Update R (Covariance Matrix)
            # Công thức gốc: R_new = R - R @ X.T @ K @ X @ R
            # Tối ưu: R_new = R - Gain @ R_XT.T
            # (Vì X @ R chính là transpose của R @ X.T)
            self.R -= Gain @ R_XT.T

            # Update Weight
            # Công thức gốc: W_new = W + R_new @ X.T @ (Y - X @ W)
            # Tối ưu: W_new = W + Gain @ (Y - X @ W)
            # (Vì Gain chính là R_new @ X.T đã tính ở trên)
            self.weight += Gain @ (Y - X @ self.weight)
    def forward(self, x, new_forward: bool = False):
        
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        if self.training:
            print("!!! NOISE IS RUNNING !!!")
        # [ADDED] Cast về dtype của weight (thường là FP16 nếu đang autocast)
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {
            'logits': logits
        }
    
    def update_task_prototype(self, prototype):
        # [MODIFIED] Lưu về CPU
        if isinstance(prototype, torch.Tensor):
            self.task_prototypes[-1] = prototype.detach().cpu()
        else:
            self.task_prototypes[-1] = prototype # Giả sử đã xử lý ở min.py

    def extend_task_prototype(self, prototype):
        # [MODIFIED] Lưu về CPU
        if isinstance(prototype, torch.Tensor):
            self.task_prototypes.append(prototype.detach().cpu())
        else:
            self.task_prototypes.append(prototype)

    def extract_feature(self, x):
        hyper_features = self.backbone(x)
        return hyper_features

    def forward_normal_fc(self, x, new_forward: bool = False):

        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # [MODIFIED] Logic ép kiểu an toàn
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype) 
        
        logits = self.normal_fc(hyper_features)['logits']
        return {
            "logits": logits
        }

    def update_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()
            self.backbone.noise_maker[j].init_weight_noise(self.task_prototypes)

    def unfreeze_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()

    def init_unfreeze(self):
        for j in range(self.backbone.layer_num):
            for param in self.backbone.noise_maker[j].parameters():
                param.requires_grad = True
            for p in self.backbone.blocks[j].norm1.parameters():
                p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
    # --- TRONG CLASS MiNbaseNet ---

    # [ADDED] Hàm điều khiển mode cho PiNoise
    def set_noise_mode(self, mode):
        if hasattr(self.backbone, 'noise_maker'):
            for m in self.backbone.noise_maker:
                m.active_task_idx = mode

    # [ADDED] Logic tìm kết quả tự tin nhất (Max Logit)
    def forward_tuna_combined(self, x):
        was_training = self.training
        self.eval()
        batch_size = x.shape[0]
        num_tasks = len(self.task_prototypes)
        
        # 1. Trích xuất Đặc trưng nền (Mode -2: Universal)
        self.set_noise_mode(-2)
        with torch.no_grad():
            feat_backbone = self.backbone(x) # [Batch, 768]
            feat_buffer = self.buffer(feat_backbone)
            logits_uni = self.forward_fc(feat_buffer)

        # 2. COSINE SIMILARITY ROUTING (Chọn Expert dựa trên không gian vector)
        selected_task_ids = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        if num_tasks > 0:
            with torch.no_grad():
                # Chuẩn hóa để tính Cosine nhanh
                feat_norm = F.normalize(feat_backbone, p=2, dim=1) # [B, D]
                all_protos = torch.stack(self.task_prototypes).to(x.device) # [N_task, D]
                all_protos_norm = F.normalize(all_protos, p=2, dim=1)
                
                # Sim matrix: [B, N_task]
                sim_matrix = torch.mm(feat_norm, all_protos_norm.t())
                # Lấy index của Task giống nhất
                selected_task_ids = sim_matrix.argmax(dim=1)

        # 3. SPECIFIC FORWARD (Chỉ chạy Expert cho mẫu tương ứng)
        best_logits_spec = torch.zeros_like(logits_uni)
        with torch.no_grad():
            unique_tasks = selected_task_ids.unique()
            for t in unique_tasks:
                mask = (selected_task_ids == t)
                self.set_noise_mode(t.item()) # Bật Expert t
                # Re-forward Backbone chỉ cho những mẫu thuộc task t
                feat_t = self.backbone(x[mask])
                l_t = self.forward_fc(self.buffer(feat_t))
                best_logits_spec[mask] = l_t

        # 4. Cộng gộp kết quả (Ensemble)
        # final_logits = logits_uni + best_logits_spec
        final_logits = best_logits_spec
        self.set_noise_mode(-2)
        if was_training: self.train()
        return {'logits': final_logits}