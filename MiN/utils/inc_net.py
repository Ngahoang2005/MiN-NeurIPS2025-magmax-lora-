import copy
import logging
import math
import numpy as np
import torch
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


class FeCAM_Manager(nn.Module):
    """
    Implement sát sườn base.py và fecam.py của source gốc.
    """
    def __init__(self, feature_dim, device, args):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.args = args
        
        # Config từ bài báo
        self.use_tukey = args.get("tukey", True)
        self.beta = args.get("beta", 0.5)
        self.alpha1 = args.get("alpha1", 1.0)
        self.alpha2 = args.get("alpha2", 1.0)
        self.use_shrink = args.get("shrink", True)
        self.use_norm_cov = args.get("norm_cov", True)
        
        # Mean (Proto)
        self.register_buffer("class_means", torch.zeros(0, feature_dim))
        
        # Covariance Storage (Inversed)
        self.class_cov_invs = {} 
        self.known_classes = 0

    def _tukeys_transform(self, x):
        """base.py dòng 173: Power transform"""
        if self.beta == 0:
            return torch.log(x)
        else:
            return torch.sign(x) * torch.pow(torch.abs(x), self.beta)

    def update_stats(self, network, data_loader):
        network.eval()
        print(f"--> [FeCAM] Updating Stats (Official Logic: CPU Collect, Tukey, Shrink, Norm)...")
        
        # 1. Collect Features to CPU (Giống _extract_vectors base.py dòng 309)
        all_feats = []
        all_labels = []
        
        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                
                # Extract
                raw_feats = network.backbone(inputs)
                
                # Move to CPU immediately (như tensor2numpy)
                all_feats.append(raw_feats.cpu())
                all_labels.append(targets.cpu())
                
        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Tukey Transform
        if self.use_tukey:
            all_feats = self._tukeys_transform(all_feats)
        
        current_classes = all_labels.unique()
        max_cls = int(current_classes.max().item()) + 1
        
        if max_cls > self.class_means.shape[0]:
            pad_size = max_cls - self.class_means.shape[0]
            self.class_means = torch.cat([self.class_means, torch.zeros(pad_size, self.feature_dim).to(self.device)], dim=0)
        
        # 2. Compute Mean & Cov per Class
        for c in current_classes:
            c_idx = int(c.item())
            idxs = (all_labels == c)
            
            # Đưa 1 class lên GPU
            vectors = all_feats[idxs].to(self.device)
            
            # Mean
            class_mean = vectors.mean(dim=0)
            self.class_means[c_idx] = class_mean
            
            # Covariance (Centered)
            x_minus_mu = vectors - class_mean
            N = vectors.shape[0]
            
            if N > 1:
                cov = (x_minus_mu.t() @ x_minus_mu) / (N - 1)
            else:
                cov = torch.eye(self.feature_dim).to(self.device) * 1e-4

            # 3. Shrinkage (base.py dòng 157)
            if self.use_shrink:
                cov = self._shrink_cov(cov)
            
            # 4. Normalize (base.py dòng 168)
            if self.use_norm_cov:
                cov = self._normalize_cov(cov)
            
            # 5. Inverse (base.py dòng 144 dùng pinv)
            inv_cov = torch.linalg.pinv(cov)
            
            self.class_cov_invs[c_idx] = inv_cov
            
            del vectors, x_minus_mu, cov, inv_cov
            
        self.known_classes = max(self.known_classes, max_cls)
        
        # Dọn rác CPU
        del all_feats, all_labels
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"--> [FeCAM] Update done. Total classes: {self.known_classes}")

    def _shrink_cov(self, cov):
        """base.py line 157"""
        diag = torch.diagonal(cov)
        diag_mean = torch.mean(diag)
        
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        
        if mask.sum() > 0:
            off_diag_mean = (off_diag * mask).sum() / mask.sum()
        else:
            off_diag_mean = torch.tensor(0.0).to(self.device)
            
        iden = torch.eye(cov.shape[0]).to(self.device)
        
        cov_ = cov + (self.alpha1 * diag_mean * iden) + (self.alpha2 * off_diag_mean * (1 - iden))
        return cov_

    def _normalize_cov(self, cov):
        """base.py line 168"""
        sd = torch.sqrt(torch.diagonal(cov)) 
        sd = torch.clamp(sd, min=1e-6)
        normalization_matrix = torch.matmul(sd.unsqueeze(1), sd.unsqueeze(0))
        cov_norm = cov / normalization_matrix
        return cov_norm

    def compute_scores(self, raw_features, active_classes, boundary_idx=0):
        """
        Logic _maha_dist và _mahalanobis trong base.py (dòng 139)
        """
        # 1. Transform input feature (nếu có tukey)
        if self.use_tukey:
            vectors = self._tukeys_transform(raw_features)
        else:
            vectors = raw_features
            
        scores = []
        distances = []
        
        for c_idx in range(active_classes):
            if c_idx not in self.class_cov_invs:
                scores.append(torch.full((raw_features.shape[0],), -1e9, device=self.device))
                distances.append(torch.full((raw_features.shape[0],), 1e9, device=self.device))
                continue
                
            inv_cov = self.class_cov_invs[c_idx]
            class_mean = self.class_means[c_idx]
            
            # [TIỂU TIẾT QUAN TRỌNG NHẤT]: base.py dòng 141
            # L2 Normalize TRƯỚC KHI trừ mean
            vectors_norm = F.normalize(vectors, p=2, dim=-1)
            mean_norm = F.normalize(class_mean, p=2, dim=-1)
            
            x_minus_mu = vectors_norm - mean_norm 
            
            # Mahalanobis calculation
            left_term = torch.matmul(x_minus_mu, inv_cov)
            mahal_diag = torch.sum(left_term * x_minus_mu, dim=1)
            
            scores.append(-mahal_diag)
            distances.append(mahal_diag)
            
        scores_stack = torch.stack(scores, dim=1)
        distances_stack = torch.stack(distances, dim=1)

        # Log bias để debug
        if boundary_idx > 0 and boundary_idx < active_classes:
            dist_old = distances_stack[:, :boundary_idx]
            dist_new = distances_stack[:, boundary_idx:]
            
            if dist_old.numel() > 0 and dist_new.numel() > 0:
                min_dist_old = dist_old.min(dim=1)[0].mean().item()
                min_dist_new = dist_new.min(dim=1)[0].mean().item()
                print(f"\r[FeCAM Check] Avg Min Dist => Old: {min_dist_old:.4f} | New: {min_dist_new:.4f} | Bias Ratio: {min_dist_old/(min_dist_new+1e-6):.2f}", end="")

        return scores_stack


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
        
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), **factory_kwargs))
        # RLS Memory Safe
        self.R_cpu = torch.eye(self.buffer_size, dtype=torch.float32) / self.gamma
        
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        
        # Init FeCAM Manager
        self.fecam = FeCAM_Manager(self.feature_dim, self.device, args)

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
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None: nn.init.constant_(new_fc.bias, 0.)
        self.normal_fc = new_fc

    def update_fecam(self, train_loader):
        self.fecam.update_stats(self, train_loader)

    def predict_combined(self, x, beta=0.5):
        """
        Combine: Analytic Logits (RLS) + FeCAM Scores (Z-score Norm)
        """
        with torch.no_grad():
            # RLS Path
            f_raw = self.backbone(x)
            f_buf = self.buffer(f_raw.to(torch.float32))
            logits_rls = self.forward_fc(f_buf) 
            # Sử dụng self.known_class (số ít) thay vì known_classes
            curr_logits_rls = logits_rls[:, :self.known_class]

            # FeCAM Path (pass active classes count)
            # Boundary index cho logging
            boundary = 0
            if self.cur_task > 0 and 'increment' in self.args:
                boundary = self.known_class - self.args['increment']
                
            # [FIXED]: self.known_class (singular)
            scores_fecam = self.fecam.compute_scores(f_raw, self.known_class, boundary_idx=boundary)
            
            # Z-Score Normalization & Fusion
            rls_mean = curr_logits_rls.mean(dim=1, keepdim=True)
            rls_std = curr_logits_rls.std(dim=1, keepdim=True) + 1e-8
            norm_rls = (curr_logits_rls - rls_mean) / rls_std
            
            fecam_mean = scores_fecam.mean(dim=1, keepdim=True)
            fecam_std = scores_fecam.std(dim=1, keepdim=True) + 1e-8
            norm_fecam = (scores_fecam - fecam_mean) / fecam_std
            
            final_logits = norm_rls * (1 - beta) + beta * norm_fecam
            
            return {'logits': final_logits}

    def forward(self, x, new_forward=False, use_fecam=False, beta=0.5):
        if use_fecam and not self.training:
            return self.predict_combined(x, beta=beta)
        if new_forward: hyper_features = self.backbone(x, new_forward=True)
        else: hyper_features = self.backbone(x)
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {'logits': logits}

    # ... Noise & GPM & RLS Fit ...
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

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        try: from torch.amp import autocast
        except ImportError: from torch.cuda.amp import autocast
        R_gpu = self.R_cpu.to(self.device)
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
            term = torch.eye(X.shape[0], device=X.device) + X @ R_gpu @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            try: K = torch.linalg.solve(term + jitter, X @ R_gpu); K = K.T
            except: K = R_gpu @ X.T @ torch.inverse(term + jitter)
            R_gpu -= K @ X @ R_gpu
            self.weight += K @ (Y - X @ self.weight)
            self.R_cpu = R_gpu.cpu()
            del term, jitter, K, X, Y, R_gpu
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