import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
# Import autocast để tắt nó trong quá trình tính toán ma trận chính xác cao
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
    """
    Lớp mở rộng đặc trưng ngẫu nhiên (Random Projection).
    """
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # [QUAN TRỌNG] Sử dụng float32 để đảm bảo độ chính xác khi tính RLS
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Ép kiểu input X về cùng kiểu với weight (float32)
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.W)


class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        
        # Các tham số cho Analytic Learning (RLS)
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        # Random Buffer
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # Khởi tạo ma trận trọng số và ma trận hiệp phương sai cho RLS
        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) # Trọng số của Analytic Classifier

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) # Ma trận hiệp phương sai đảo (Inverse Covariance Matrix)

        # Normal FC
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        
        # --- [FeCAM STORAGE] ---
        # 1. Class Means (Transformed space)
        self.register_buffer("class_means", torch.zeros(0, self.feature_dim)) 
        
        # 2. Common Covariance (Accumulated Weighted Average)
        self.register_buffer("common_cov", torch.zeros(self.feature_dim, self.feature_dim))
        
        # 3. Inference Matrix (Correlation Normalized Inverse)
        self.register_buffer("common_corr_inv", torch.eye(self.feature_dim))
        self.register_buffer("diag_std", torch.ones(self.feature_dim)) # Để normalize feature lúc inference
        
        # Biến đếm số lượng class cũ để tính trọng số update Covariance
        self.fecam_known_classes = 0 

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        
        # Tạo mới Normal FC cho task hiện tại
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
    # [FeCAM IMPLEMENTATION]
    # =========================================================================

    def _tukey_transform(self, x):
        """Eq.9: Power transform (Square root)"""
        return torch.sqrt(torch.clamp(x, min=1e-8))

    def update_fecam_stats(self, train_loader):
        """
        FeCAM Update Logic:
        1. Tukey Transform
        2. Compute Task Common Covariance
        3. Weighted Update Global Covariance
        4. Shrinkage & Normalization
        """
        self.eval()
        print(f"--> [FeCAM] Updating Stats for Task {self.cur_task}...")
        
        all_feats = []
        all_labels = []
        
        # 1. Collect & Transform Features
        with torch.no_grad():
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                
                # Raw backbone features
                raw_feats = self.backbone(inputs)
                
                # Tukey Transform (Eq.9)
                trans_feats = self._tukey_transform(raw_feats)
                
                all_feats.append(trans_feats)
                all_labels.append(targets.to(self.device))
                
        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 2. Update Class Means
        current_classes = all_labels.unique()
        max_cls = int(current_classes.max().item()) + 1
        
        # Mở rộng buffer class_means nếu cần
        if max_cls > self.class_means.shape[0]:
            pad_size = max_cls - self.class_means.shape[0]
            self.class_means = torch.cat([self.class_means, torch.zeros(pad_size, self.feature_dim).to(self.device)], dim=0)
            
        for c in current_classes:
            idxs = (all_labels == c)
            # Mean calculated on transformed features
            self.class_means[int(c.item())] = all_feats[idxs].mean(dim=0)
            
        # 3. Compute Task Common Covariance (Sigma^t)
        # Center features theo mean của chính nó
        centered_feats = torch.empty_like(all_feats)
        for c in current_classes:
            idxs = (all_labels == c)
            centered_feats[idxs] = all_feats[idxs] - self.class_means[int(c.item())]
            
        N = all_feats.shape[0]
        # Covariance của task hiện tại
        task_cov = (centered_feats.t() @ centered_feats) / (N - 1 + 1e-6)
        
        # 4. Incremental Update Global Covariance
        # Weighted average based on number of classes
        # Formula: Sigma_new = (N_old * Sigma_old + N_new * Sigma_new) / N_total
        # Lưu ý: FeCAM dùng số lượng class để weight
        
        num_new_classes = len(current_classes)
        total_classes = self.fecam_known_classes + num_new_classes
        
        if self.fecam_known_classes == 0:
            self.common_cov = task_cov
        else:
            w_old = self.fecam_known_classes / total_classes
            w_new = num_new_classes / total_classes
            self.common_cov = self.common_cov * w_old + task_cov * w_new
            
        self.fecam_known_classes = total_classes
        
        # 5. Compute Inference Matrix (Shrinkage + Norm)
        self._compute_inference_matrix()
        print(f"--> [FeCAM] Update Complete. Total classes tracked: {self.fecam_known_classes}")

    def _compute_inference_matrix(self):
        """
        Shrinkage (Eq.8) & Correlation Normalization (Eq.7)
        """
        Sigma = self.common_cov
        D = self.feature_dim
        
        # --- Shrinkage (Eq.8) ---
        # V1: Average diagonal variance
        V1 = torch.trace(Sigma) / D
        
        # V2: Average off-diagonal covariance
        mask = ~torch.eye(D, dtype=torch.bool, device=self.device)
        V2 = Sigma[mask].mean()
        
        # Gamma coefficients (Paper recommends 1.0 for many-shot)
        gamma1, gamma2 = 1.0, 1.0
        
        Identity = torch.eye(D, device=self.device)
        Sigma_s = Sigma + gamma1 * V1 * Identity + gamma2 * V2 * (1 - Identity)
        
        # --- Correlation Normalization (Eq.7) ---
        # Chuẩn hóa covariance thành correlation matrix
        # Điều này giúp distance comparable giữa các task
        diag_std = torch.sqrt(torch.diag(Sigma_s)) # [D]
        self.diag_std = diag_std # Lưu lại để normalize feature input
        
        outer_std = torch.outer(diag_std, diag_std)
        Sigma_hat = Sigma_s / (outer_std + 1e-8)
        
        # Inverse & Save
        # Dùng pinverse để tránh singular matrix
        self.common_corr_inv = torch.linalg.pinv(Sigma_hat)

    def predict_combined(self, x, beta=1.0):
        """
        Combine Analytic Logits (RLS) + FeCAM Scores (Mahalanobis)
        """
        # --- 1. Analytic Logits (RLS) ---
        with torch.no_grad():
            f_raw = self.backbone(x)
            f_buf = self.buffer(f_raw.to(torch.float32))
            logits_rls = self.forward_fc(f_buf) # [B, known_class]
            
            # --- 2. FeCAM Scores ---
            # Transform input y hệt lúc train
            f_trans = self._tukey_transform(f_raw) # [B, D]
            
            # Normalize feature theo Correlation Matrix logic (chia cho std dev)
            # x_norm = x / sigma
            f_norm = f_trans / (self.diag_std + 1e-8) # [B, D]
            
            # Lấy Means của các class đã học
            active_classes = self.known_class
            means = self.class_means[:active_classes] # [C, D]
            
            # Normalize means
            means_norm = means / (self.diag_std + 1e-8) # [C, D]
            
            # Tính Mahalanobis (Optimized)
            # Inv = self.common_corr_inv
            # Score = 2 * x^T Inv mu - mu^T Inv mu
            
            # Pre-compute mu terms
            # [C, D] @ [D, D]
            mu_proj = means_norm @ self.common_corr_inv
            # [C]
            term_mu = torch.sum(mu_proj * means_norm, dim=1).unsqueeze(0) # [1, C]
            
            # Compute Cross term
            # [B, D] @ [D, D] @ [D, C] -> [B, D] @ [D, C]
            term_cross = (f_norm @ self.common_corr_inv) @ means_norm.t() # [B, C]
            
            scores_fecam = 2 * term_cross - term_mu
            
            # --- 3. Combine ---
            # Đảm bảo logits_rls khớp size (phòng khi buffer lớn hơn)
            curr_logits_rls = logits_rls[:, :active_classes]
            
            final_logits = curr_logits_rls + beta * scores_fecam
            
            return {
                'logits': final_logits,
                'logits_rls': curr_logits_rls,
                'logits_fecam': scores_fecam
            }

    def forward(self, x, new_forward=False, use_fecam=False, beta=0.8):
        # Chế độ Inference với FeCAM
        if use_fecam and not self.training:
            return self.predict_combined(x, beta=beta)
        
        # Chế độ Train/Eval thường
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {'logits': logits}

    # =========================================================================
    # [MAGMAX & NOISE CONTROL SECTION]
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

    # =========================================================================
    # [ANALYTIC LEARNING (RLS) SECTION]
    # =========================================================================

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