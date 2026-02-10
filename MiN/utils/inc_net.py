import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
from torch.cuda.amp import autocast 
from torch.distributions.multivariate_normal import MultivariateNormal

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

        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

        # [NEW] Thêm biến để lưu Mean và Covariance
        self.class_mean = {}
        self.class_cov = {}

    def update_fc(self, nb_classes):
        self.cur_task += 1
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
            self.normal_fc = new_fc
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None:
                nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    # =========================================================================
    # [NEW SECTION] STATISTICS & PSEUDO SAMPLING
    # =========================================================================
    def compute_class_statistics(self, data_loader):
        """
        Tính toán Mean và Covariance cho các class trong task hiện tại.
        Gọi hàm này sau khi train xong mỗi task.
        """
        print(f"--> [IncNet] Computing Mean & Covariance for Task {self.cur_task}...")
        self.eval()
        
        # Dictionary tạm để gom features theo class
        class_features = {}
        
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Trích xuất đặc trưng từ backbone (chưa qua buffer)
                features = self.backbone(inputs)
                
                for f, t in zip(features, targets):
                    label = t.item()
                    if label not in class_features:
                        class_features[label] = []
                    class_features[label].append(f.cpu())

        # Tính Mean và Cov
        for label, feats in class_features.items():
            feats = torch.stack(feats).to(self.device) # [N, Feature_Dim]
            
            # Tính Mean
            mean = torch.mean(feats, dim=0)
            self.class_mean[label] = mean
            
            # Tính Covariance
            # Lưu ý: Nếu N nhỏ feature_dim, dùng shrinked cov hoặc diagonal
            if feats.shape[0] > 1:
                cov = torch.cov(feats.T) # [Feature_Dim, Feature_Dim]
                # Thêm nhiễu nhỏ vào đường chéo để tránh lỗi Singular Matrix khi sampling
                cov = cov + torch.eye(cov.shape[0], device=self.device) * 1e-4
            else:
                cov = torch.eye(feats.shape[1], device=self.device) * 1e-4
                
            self.class_cov[label] = cov
            
        print(f"--> [IncNet] Updated stats for {len(class_features)} classes.")

    def generate_pseudo_features(self, num_samples_per_class=1):
        """
        Sinh mẫu giả từ các class CŨ (các class không thuộc task hiện tại).
        Trả về (Pseudo_Inputs, Pseudo_Labels) đã qua Buffer projection.
        """
        pseudo_features = []
        pseudo_labels = []
        
        # Chỉ sinh mẫu cho các class đã học TRƯỚC task này
        # (Giả sử các class mới nhất chưa cần replay ngay lúc training task đó)
        # Hoặc sinh cho tất cả các class đã có trong self.class_mean
        if len(self.class_mean) == 0:
            return None, None

        for label, mean in self.class_mean.items():
            cov = self.class_cov[label]
            
            # Feature Replay bằng Multivariate Normal
            # Cần try-catch vì đôi khi Cov matrix bị lỗi số học
            try:
                dist = MultivariateNormal(mean, covariance_matrix=cov)
                samples = dist.sample((num_samples_per_class,))
            except RuntimeError:
                # Fallback nếu lỗi: dùng Mean + Standard Normal Noise
                samples = mean.unsqueeze(0) + torch.randn(num_samples_per_class, mean.shape[0], device=self.device) * 0.1
                
            pseudo_features.append(samples)
            pseudo_labels.append(torch.full((num_samples_per_class,), label, device=self.device, dtype=torch.long))
            
        if len(pseudo_features) == 0:
            return None, None
            
        X_pseudo = torch.cat(pseudo_features, dim=0)
        Y_pseudo = torch.cat(pseudo_labels, dim=0)
        
        # Quan trọng: Mẫu giả này là raw feature từ backbone.
        # Hàm fit() sẽ tự gọi self.buffer(X), nên ta trả về raw feature ở đây là đúng.
        return X_pseudo, Y_pseudo
    
    # =========================================================================
    
    def update_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    def after_task_magmax_merge(self):
        print(f"--> [IncNet] Task {self.cur_task}: Triggering Parameter-wise MagMax Merging...")
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_incremental()

    def init_unfreeze(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_task_0()
            if hasattr(self.backbone.blocks[j], 'norm1'):
                for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            if hasattr(self.backbone.blocks[j], 'norm2'):
                for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
                
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters(): p.requires_grad = True

    def forward_fc(self, features):
        features = features.to(self.weight.dtype) 
        return features @ self.weight

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        try:
            from torch.amp import autocast
        except ImportError:
            from torch.cuda.amp import autocast

        with autocast('cuda', enabled=False):
            X = self.backbone(X).float() 
            X = self.buffer(X) 
            
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.weight.shape[1]:
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size), device=Y.device)
                Y = torch.cat((Y, tail), dim=1)

            term = torch.eye(X.shape[0], device=X.device) + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            try:
                K = torch.linalg.solve(term + jitter, X @ self.R)
                K = K.T 
            except:
                K = self.R @ X.T @ torch.inverse(term + jitter)

            self.R -= K @ X @ self.R
            self.weight += K @ (Y - X @ self.weight)
            
            del term, jitter, K, X, Y

    def forward(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {'logits': logits}

    def extract_feature(self, x):
        return self.backbone(x)

    def forward_normal_fc(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = self.buffer(hyper_features.to(self.buffer.weight.dtype))
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}

    def collect_projections(self, mode='threshold', val=0.95):
        print(f"--> [IncNet] Collecting Projections (Mode: {mode}, Val: {val})...")
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].compute_projection_matrix(mode=mode, val=val)

    def apply_gpm_to_grads(self, scale=1.0):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].apply_gradient_projection(scale=scale)