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
from torch.cuda.amp import autocast 

class BaseIncNet(nn.Module):
    # ... (Giữ nguyên phần đầu) ...
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
        start = self.known_class
        end = self.known_class + nb_classes
        if not hasattr(self, 'task_class_indices'):
            self.task_class_indices = []
        self.task_class_indices.append(list(range(start, end)))
    @staticmethod
    def generate_fc(in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

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
        self.task_prototypes = []

        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)
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
        
        # [QUAN TRỌNG] Thêm H, Hy để tránh lỗi AttributeError trong Trainer
        self.register_buffer("H", torch.zeros((self.buffer_size, self.buffer_size), **factory_kwargs))
        self.register_buffer("Hy", torch.zeros((self.buffer_size, 0), **factory_kwargs))

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
            # Backbone sẽ dùng mode hiện tại (đã được set ở Trainer) để trích xuất đặc trưng
            X = self.backbone(X).float()
            X = self.buffer(X)
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            num_targets = Y.shape[1]
            if num_targets > self.out_features:
                increment_size = num_targets - self.out_features
                tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device, dtype=self.weight.dtype)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.out_features:
                increment_size = self.out_features - num_targets
                tail = torch.zeros((Y.shape[0], increment_size), device=Y.device, dtype=Y.dtype)
                Y = torch.cat((Y, tail), dim=1)

            I = torch.eye(X.shape[0]).to(X)
            term = I + X @ self.R @ X.T
            term.diagonal().add_(1.0 + 1e-6)
            K_inv = torch.inverse(term)
            Gain = R_XT = self.R @ X.T @ K_inv # Combine steps for clarity
            
            self.R -= Gain @ (self.R @ X.T).T
            self.weight += Gain @ (Y - X @ self.weight)

    def forward(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {'logits': logits}
    
    def update_task_prototype(self, prototype):
        if isinstance(prototype, torch.Tensor):
            self.task_prototypes[-1] = prototype.detach().cpu()
        else:
            self.task_prototypes[-1] = prototype

    def extend_task_prototype(self, prototype):
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
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype) 
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}

    def update_noise(self):
        # Lấy Mean của các class proto để init noise (đơn giản hóa thành 1 vector/task)
        task_means = []
        if len(self.task_prototypes) > 0:
            for p in self.task_prototypes:
                # p là [Num_Class, Dim], mean lại thành [Dim]
                task_means.append(p.mean(dim=0))
        
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()
            self.backbone.noise_maker[j].init_weight_noise(task_means)

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

    def set_noise_mode(self, mode):
        if hasattr(self.backbone, 'noise_maker'):
            for m in self.backbone.noise_maker:
                m.active_task_idx = mode

    # [FIXED] Dùng mode -2 cho Uni khi Infer và kết hợp với Expert, dùng expert để tính prototype
    # [FIXED LOGIC] Single Pass Expert Routing + Logit Masking
    def forward_tuna_combined(self, x):
        was_training = self.training
        self.eval()
        batch_size = x.shape[0]
        num_tasks = len(self.task_prototypes)
        
        self.set_noise_mode(-2)
        with torch.no_grad():
            feat_noisy_uni = self.backbone(x)
            logits_uni = self.forward_fc(self.buffer(feat_noisy_uni))

        # BƯỚC 2: LẤY FEATURE SẠCH (MODE -3) ĐỂ ROUTING
        # Phải forward thêm 1 lần (hoặc lấy feature từ bước trước nếu kiến trúc cho phép)
        # để đảm bảo so sánh "Sạch vs Sạch"
        self.set_noise_mode(-3)
        with torch.no_grad():
            feat_clean = self.backbone(x) 
            feat_norm = F.normalize(feat_clean, p=2, dim=1)
            
            # Tính Similarity với Prototype (cũng tính ở -3)
            task_scores = []
            for t_idx, protos in enumerate(self.task_prototypes):
                p_norm = F.normalize(protos.to(x.device), p=2, dim=1)
                sim_t, _ = torch.mm(feat_norm, p_norm.t()).max(dim=1)
                task_scores.append(sim_t)
            
            selected_task_ids = torch.stack(task_scores, dim=1).argmax(dim=1)

        # BƯỚC 3: LẤY LOGIT EXPERT (SPECIFIC)
        best_logits_spec = torch.zeros_like(logits_uni)
        with torch.no_grad():
            for t in selected_task_ids.unique():
                mask = (selected_task_ids == t)
                self.set_noise_mode(t.item())
                # ... forward expert và masking như cũ ...
                best_logits_spec[mask] = masked_l

        # BƯỚC 4: ENSEMBLE
        # Bây giờ bạn có Uni chuẩn (-2) và Spec chuẩn (Mode t)
        final_logits = best_logits_spec 
        
        self.set_noise_mode(-2)
        return {'logits': final_logits}