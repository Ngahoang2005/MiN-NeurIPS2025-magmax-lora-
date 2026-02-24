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
    def fit(self, X: torch.Tensor, Y: torch.Tensor, chunk_size=2048) -> None:
        """
        Tối ưu hóa Analytic Learning (Chunking RLS):
        1. Sử dụng Accumulation (Cộng dồn) để tránh OOM khi tính X^T * X trên dataset lớn.
        2. Sử dụng torch.linalg.solve thay vì torch.inverse (Nhanh hơn & Ổn định hơn).
        """
        # [QUAN TRỌNG] Tắt Mixed Precision để đảm bảo độ chính xác ma trận
        with autocast(enabled=False):
            
            # 1. Chuẩn bị dữ liệu (Float32)
            X = X.float().to(self.device)
            Y = Y.float().to(self.device)
            
            # 2. Mở rộng Classifier nếu có class mới
            num_targets = Y.shape[1]
            
            # Nếu chưa có weight (lần đầu fit), khởi tạo
            if self.weight.shape[1] == 0:
                # Tạm tính feature dim sau khi qua buffer
                dummy_feat = self.backbone(X[0:2]).float()
                dummy_feat = self.buffer(dummy_feat)
                feat_dim = dummy_feat.shape[1]
                
                self.weight = torch.zeros((feat_dim, num_targets), device=self.device, dtype=torch.float32)
                
            elif num_targets > self.weight.shape[1]:
                # Mở rộng weight cũ (Padding 0 cho class mới)
                increment = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment), device=self.device, dtype=torch.float32)
                self.weight = torch.cat((self.weight, tail), dim=1)

            # 3. Tính toán Ma trận A (Autocorrelation) và B (Cross-correlation)
            # A = X^T * X + lambda * I
            # B = X^T * Y
            
            N = X.shape[0]
            feat_dim = self.weight.shape[0]
            
            A = torch.zeros((feat_dim, feat_dim), device=self.device, dtype=torch.float32)
            B = torch.zeros((feat_dim, num_targets), device=self.device, dtype=torch.float32)
            
            # Kỹ thuật Chunking: Duyệt qua từng batch nhỏ
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                
                # Lấy raw images
                x_batch = X[start:end] 
                y_batch = Y[start:end] 
                
                # Extract features qua backbone + buffer
                features = self.backbone(x_batch).float()
                features = self.buffer(features) # Qua Random Projection
                
                # Cộng dồn
                A += features.T @ features
                B += features.T @ y_batch
                
                del features, x_batch, y_batch 

            # 4. Áp dụng Regularization (Ridge)
            I = torch.eye(feat_dim, device=self.device, dtype=torch.float32)
            A += self.gamma * I 

            # 5. Giải hệ phương trình tuyến tính A * W = B
            try:
                # linalg.solve tự động chọn thuật toán (Cholesky/LU) tối ưu
                W_solution = torch.linalg.solve(A, B)
            except RuntimeError:
                # Fallback dùng Pseudo-Inverse nếu ma trận suy biến
                W_solution = torch.linalg.pinv(A) @ B
            
            # 6. Cập nhật Weight
            self.weight = W_solution
            
            del A, B, I, X, Y
            torch.cuda.empty_cache()
    def forward(self, x, new_forward: bool = False):
        
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
       
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
    def forward_with_ib(self, x):
        kl_losses = []
        
        x = self.backbone.patch_embed(x)
        if hasattr(self.backbone, '_pos_embed'):
            x = self.backbone._pos_embed(x)
        else:
            if self.backbone.pos_embed is not None:
                x = x + self.backbone.pos_embed
            x = self.backbone.pos_drop(x)

        for i, block in enumerate(self.backbone.blocks):
            x = block(x) 
            if hasattr(self.backbone, 'noise_maker'):
                # [QUAN TRỌNG]: Bật return_kl = True
                x, kl = self.backbone.noise_maker[i](x, return_kl=True)
                kl_losses.append(kl)
        
        if hasattr(self.backbone, 'norm'):
            x = self.backbone.norm(x)

        if x.dim() == 3: x = x[:, 0]

        x = self.buffer(x.to(self.buffer.weight.dtype))
        x = x.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(x)['logits']
        
        # Gom KL Loss của toàn bộ 12 layers
        total_kl = sum(kl_losses) if kl_losses else torch.tensor(0.0, device=self.device)
        
        return logits, total_kl