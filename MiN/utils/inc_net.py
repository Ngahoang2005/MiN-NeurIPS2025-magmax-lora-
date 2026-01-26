import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from backbones.pretrained_backbone import get_pretrained_backbone 
from backbones.linears import SimpleLinear
from torch.nn import functional as F
import gc

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

class BaseIncNet(nn.Module):
    def __init__(self, args: dict):
        super(BaseIncNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        
        # [QUAN TRỌNG]: Tắt gradient backbone ngay lập tức để tiết kiệm VRAM khởi tạo
        for param in self.backbone.parameters():
            param.requires_grad = False
            
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
        
        # [FIX OOM]: Bắt buộc dùng Float32.
        # Nếu dùng Double: 768 * 8192 * 8 bytes = 50MB (Không lớn, nhưng tích tiểu thành đại)
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        # Khởi tạo Kaiming để phân phối giá trị tốt
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        
        self.register_buffer("weight", self.W)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight.dtype) 
        return F.relu(X @ self.W)

class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        # Dọn rác trước khi khởi tạo
        gc.collect(); torch.cuda.empty_cache()
        
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        self.gamma = args['gamma']
        
        # [FIX OOM CỰC MẠNH]: Giảm buffer size nếu đang để 16384
        # Khuyến nghị: 8192 (hoặc 4096 nếu GPU quá yếu)
        self.buffer_size = args['buffer_size'] 
        print(f"📉 [MiNbaseNet] Initializing with Buffer Size: {self.buffer_size}")
        
        self.feature_dim = self.backbone.out_dim 
        
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)
        
        # [THỦ PHẠM 2GB]: RLS Matrices
        # Thay vì tạo torch.eye (Full Matrix), ta dùng Sparse hoặc tạo khi cần.
        # Nhưng để đơn giản và nhanh, ta ép kiểu Float32.
        # 16384^2 * 4 bytes = 1 GB (Chấp nhận được). 
        # Nếu là Double sẽ là 2GB.
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        
        # Weight Matrix (Khởi tạo rỗng)
        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) 

        # Covariance Matrix (Identity / Gamma)
        # Tạo Identity trực tiếp trên GPU để tránh copy từ CPU
        print("Creating RLS Covariance Matrix (R)...")
        R = torch.eye(self.buffer_size, **factory_kwargs)
        R.div_(self.gamma) # In-place division để tiết kiệm RAM tạm
        self.register_buffer("R", R) 
        print(f"✅ R Matrix Created: {R.shape} | {R.element_size() * R.numel() / 1e9:.2f} GB")
        
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        
        # Tắt gradient toàn bộ mạng lúc đầu để tránh rò rỉ
        for p in self.parameters():
            p.requires_grad = False

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        if self.cur_task > 0:
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False).float()
        else:
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True).float()
            
        if self.normal_fc is not None:
            old_nb = self.normal_fc.out_features
            with torch.no_grad():
                new_fc.weight[:old_nb] = self.normal_fc.weight.data
                nn.init.constant_(new_fc.weight[old_nb:], 0.)
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None: nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    def update_noise(self):
        if hasattr(self.backbone, 'noise_maker'):
            print(f"--> [IncNet] Expanding BiLoRA Noise for Task {self.cur_task}")
            for j in range(len(self.backbone.noise_maker)):
                self.backbone.noise_maker[j].expand_new_task(self.cur_task)

    def after_task_magmax_merge(self):
        if hasattr(self.backbone, 'noise_maker'):
            for j in range(len(self.backbone.noise_maker)):
                 self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        if hasattr(self.backbone, 'noise_maker'):
            for j in range(len(self.backbone.noise_maker)):
                for param in self.backbone.noise_maker[j].parameters():
                    param.requires_grad = True

    def init_unfreeze(self):
        self.unfreeze_noise()
        if hasattr(self.backbone, 'blocks'):
            for block in self.backbone.blocks:
                if hasattr(block, 'norm1'): 
                    for p in block.norm1.parameters(): p.requires_grad = True
                if hasattr(block, 'norm2'):
                    for p in block.norm2.parameters(): p.requires_grad = True
        if hasattr(self.backbone, 'norm'):
            for p in self.backbone.norm.parameters(): p.requires_grad = True

    def forward_fc(self, features):
        features = features.to(self.weight.dtype)
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """RLS Fit: Optimized for Memory"""
        old_training_state = self.training
        self.eval() 
        try:
            with autocast('cuda', enabled=True): 
                X_feat = self.backbone(X)
            
            with autocast('cuda', enabled=False):
                X_feat = X_feat.detach().float()
                X_proj = self.buffer(X_feat)
                del X_feat 
                
                device = self.weight.device
                X_final = X_proj.to(device)
                Y = Y.to(device).float()

                # Expand weights
                num_targets = Y.shape[1]
                if num_targets > self.weight.shape[1]:
                    increment_size = num_targets - self.weight.shape[1]
                    tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                    self.weight = torch.cat((self.weight, tail), dim=1)
                elif num_targets < self.weight.shape[1]:
                    increment_size = self.weight.shape[1] - num_targets
                    tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                    Y = torch.cat((Y, tail), dim=1)

                # RLS Algorithm (Step by Step Cleanup)
                P = self.R @ X_final.T
                term = X_final @ P
                term.diagonal().add_(1.0) 
                term = 0.5 * (term + term.T)
                
                try:
                    K = torch.linalg.inv(term)
                except RuntimeError:
                    print("⚠️ GPU OOM, switching to CPU for Inverse...")
                    K = torch.linalg.inv(term.cpu()).to(device)
                del term 
                
                P_K = P @ K 
                self.R -= P_K @ P.T
                del P 
                
                residual = Y - (X_final @ self.weight)
                self.weight += P_K @ residual
                
                del X_final, Y, K, P_K, residual
                torch.cuda.empty_cache()
        finally:
            self.train(old_training_state)

    def forward(self, x, new_forward: bool = False):
        hyper_features = self.backbone(x)
        hyper_features = hyper_features.float() # Đảm bảo Float32
        proj_features = self.buffer(hyper_features)
        logits = self.forward_fc(proj_features)
        return {'logits': logits}

    def forward_normal_fc(self, x, new_forward: bool = False):
        hyper_features = self.backbone(x)
        hyper_features = hyper_features.float()
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}