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
        factory_kwargs = {"device": device, "dtype": torch.double}
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
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
        factory_kwargs = {"device": self.device, "dtype": torch.double}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

        self.Pinoise_list = nn.ModuleList()
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

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
        # [NEW] TỐI ƯU CỰC MẠNH: Dùng linalg.solve thay vì inverse
        X = self.buffer(self.backbone(X))
        X, Y = X.to(self.weight), Y.to(self.weight)

        num_targets = Y.shape[1]
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
            self.weight = torch.cat((self.weight, tail), dim=1)
        elif num_targets < self.out_features:
            increment_size = self.out_features - num_targets
            tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
            Y = torch.cat((Y, tail), dim=1)

        # Tính toán ma trận trung gian, dùng solve nhanh và chống OOM
        term = torch.eye(X.shape[0], device=X.device, dtype=X.dtype) + X @ self.R @ X.T
        try:
            K = torch.linalg.solve(term, X @ self.R).T
        except RuntimeError:
            K = self.R @ X.T @ torch.linalg.pinv(term)
            
        self.R -= K @ X @ self.R
        self.weight += K @ (Y - X @ self.weight)

    def forward(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {'logits': logits}
        
    def update_task_prototype(self, prototype):
        self.task_prototypes[-1] = prototype

    def extend_task_prototype(self, prototype):
        self.task_prototypes.append(prototype)

    def extract_feature(self, x):
        return self.backbone(x)

    def forward_normal_fc(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(torch.float32)
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}

    # [NEW] Thêm hàm Forward thu thập KL Loss 
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
                x, kl = self.backbone.noise_maker[i](x, return_kl=True)
                kl_losses.append(kl)
                
        if hasattr(self.backbone, 'norm'):
            x = self.backbone.norm(x)
            
        if x.dim() == 3: x = x[:, 0]
        
        hyper_features = self.buffer(x)
        logits = self.normal_fc(hyper_features.float())['logits']
        total_kl = sum(kl_losses) if kl_losses else torch.tensor(0.0, device=x.device)
        return logits, total_kl

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