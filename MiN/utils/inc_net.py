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
    def __init__(self, feature_dim, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        
        # Buffer lưu trữ Mean (nhỏ, để trên GPU ok)
        self.register_buffer("class_means", torch.zeros(0, feature_dim))
        
        # Dictionary lưu các ma trận nghịch đảo (mỗi ma trận ~2MB, 100 class ~200MB -> GPU chịu được)
        self.class_cov_invs = {} 
        self.class_diags = {}    
        self.known_classes = 0

    def update_stats(self, network, data_loader):
        """
        [GPU STREAMING]: Tính toán trực tiếp trên GPU nhưng KHÔNG lưu feature.
        Cơ chế: Load Batch -> Update Sum/Scatter -> DEL Batch.
        Giữ VRAM luôn sạch sẽ.
        """
        network.eval()
        print(f"--> [FeCAM] Updating Stats (GPU Streaming - No Garbage)...")
        
        # Dictionary tạm để tích lũy thống kê
        # stats[c] = {'sum': ..., 'scatter': ..., 'count': ...}
        stats = {}
        
        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 1. Forward backbone
                raw_feats = network.backbone(inputs) # [Batch, D]
                
                # 2. Phân loại theo class trong batch hiện tại
                unique_classes = targets.unique()
                
                for c in unique_classes:
                    c = c.item()
                    idxs = (targets == c)
                    # Lấy feature của class c trong batch này
                    features_c = raw_feats[idxs] 
                    
                    if c not in stats:
                        # Khởi tạo bộ đếm trên GPU (dùng double để cộng dồn chính xác)
                        stats[c] = {
                            'sum': torch.zeros(self.feature_dim, device=self.device, dtype=torch.double),
                            'scatter': torch.zeros((self.feature_dim, self.feature_dim), device=self.device, dtype=torch.double),
                            'count': 0
                        }
                    
                    # 3. Tích lũy (Streaming Step)
                    # Sum: sigma x
                    stats[c]['sum'] += features_c.sum(dim=0).double()
                    # Scatter: sigma (x * x.T) = features.T @ features
                    # Phép nhân này tốn bộ nhớ tạm thời, nhưng vì features_c nhỏ (chỉ 1 phần batch) nên an toàn
                    stats[c]['scatter'] += (features_c.t().double() @ features_c.double())
                    stats[c]['count'] += features_c.shape[0]
                
                # [QUAN TRỌNG]: Dọn rác ngay lập tức sau mỗi batch
                del raw_feats, inputs, targets, features_c
                # Ép giải phóng bộ nhớ đệm nếu cần (tùy chọn, có thể làm chậm nhưng an toàn)
                # torch.cuda.empty_cache() 
        
        # 4. Tính toán Mean và Covariance từ bộ tích lũy
        current_classes = sorted(list(stats.keys()))
        max_cls = max(current_classes) + 1 if current_classes else 0
        
        # Expand buffer
        if max_cls > self.class_means.shape[0]:
            pad_size = max_cls - self.class_means.shape[0]
            self.class_means = torch.cat([self.class_means, torch.zeros(pad_size, self.feature_dim).to(self.device)], dim=0)
            
        for c in current_classes:
            N = stats[c]['count']
            sum_x = stats[c]['sum'].float()     # Chuyển về float32
            scatter_x = stats[c]['scatter'].float()
            
            # Mean = Sum / N
            mean = sum_x / N
            self.class_means[c] = mean
            
            # Covariance = (Scatter - N * mean * mean.T) / (N - 1)
            # Công thức này tương đương với centered covariance
            if N > 1:
                # outer_mean cũng nhỏ (768x768)
                outer_mean = torch.outer(mean, mean)
                cov = (scatter_x - N * outer_mean) / (N - 1)
            else:
                cov = torch.eye(self.feature_dim).to(self.device) * 1e-3
                
            # Shrinkage & Norm
            inv_c, diag_c = self._shrink_and_normalize(cov)
            
            self.class_cov_invs[c] = inv_c
            self.class_diags[c] = diag_c
            
            # Dọn rác biến tạm
            del cov, inv_c, diag_c, sum_x, scatter_x
            
        self.known_classes = max(self.known_classes, max_cls)
        
        # Xóa toàn bộ stats
        del stats
        torch.cuda.empty_cache() # Dọn sạch sẽ để chuẩn bị cho task sau
        
        print(f"--> [FeCAM] Update done on GPU. Total classes: {self.known_classes}")

    def _shrink_and_normalize(self, sigma):
        D = self.feature_dim
        V1 = torch.trace(sigma) / D
        mask = ~torch.eye(D, dtype=torch.bool, device=self.device)
        V2 = sigma[mask].mean()
        
        gamma = 1.0 
        Identity = torch.eye(D, device=self.device)
        Sigma_s = sigma + gamma * (V1 * Identity + V2 * (1 - Identity))
        
        diag_std = torch.sqrt(torch.diag(Sigma_s)) 
        diag_std = torch.clamp(diag_std, min=1e-6)
        
        outer_std = torch.outer(diag_std, diag_std)
        Sigma_hat = Sigma_s / outer_std 
        
        inv_matrix = torch.linalg.pinv(Sigma_hat)
        
        return inv_matrix, diag_std

    def compute_scores(self, raw_features, active_classes, boundary_idx=0):
        scores = []
        distances = []
        for c_idx in range(active_classes):
            if c_idx not in self.class_cov_invs:
                scores.append(torch.full((raw_features.shape[0],), -1e9, device=self.device))
                distances.append(torch.full((raw_features.shape[0],), 1e9, device=self.device))
                continue
                
            inv_c = self.class_cov_invs[c_idx]
            diag_c = self.class_diags[c_idx]
            mean_c = self.class_means[c_idx]
            
            x_norm = raw_features / diag_c.unsqueeze(0)
            mu_norm = mean_c / diag_c
            diff = x_norm - mu_norm.unsqueeze(0)
            
            temp = diff @ inv_c
            dist_sq = torch.sum(temp * diff, dim=1) 
            scores.append(-dist_sq) 
            distances.append(dist_sq)
            
        scores_stack = torch.stack(scores, dim=1)
        distances_stack = torch.stack(distances, dim=1)

        if boundary_idx > 0 and boundary_idx < active_classes:
            dist_old = distances_stack[:, :boundary_idx]
            dist_new = distances_stack[:, boundary_idx:]
            
            if dist_old.numel() > 0 and dist_new.numel() > 0:
                min_dist_old = dist_old.min(dim=1)[0].mean().item()
                min_dist_new = dist_new.min(dim=1)[0].mean().item()
                print(f"\r[FeCAM Check] Avg Min Dist => Old: {min_dist_old:.2f} | New: {min_dist_new:.2f} | Bias Ratio: {min_dist_old/(min_dist_new+1e-6):.2f}", end="")

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
        # R vẫn để trên GPU vì user yêu cầu, nhưng nên cẩn thận
        self.register_buffer("R", torch.eye(self.buffer_size, **factory_kwargs) / self.gamma)

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        self.fecam = FeCAM_Manager(self.feature_dim, self.device)

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

    def update_fecam_stats(self, train_loader):
        self.fecam.update_stats(self, train_loader)

    def predict_combined(self, x, beta=0.5):
        with torch.no_grad():
            f_raw = self.backbone(x)
            f_buf = self.buffer(f_raw.to(torch.float32))
            logits_rls = self.forward_fc(f_buf) 
            
            active_classes = self.known_class
            boundary = 0
            if self.cur_task > 0 and 'increment' in self.args:
                boundary = self.known_class - self.args['increment']

            scores_fecam = self.fecam.compute_scores(f_raw, active_classes, boundary_idx=boundary)
            curr_logits_rls = logits_rls[:, :active_classes]

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
        
        # Xóa cache trước khi tính toán nặng
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
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

            # Phép tính này rất nặng, đảm bảo xóa biến ngay
            term = torch.eye(X.shape[0], device=X.device) + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            try: K = torch.linalg.solve(term + jitter, X @ self.R); K = K.T
            except: K = self.R @ X.T @ torch.inverse(term + jitter)
            
            self.R -= K @ X @ self.R
            self.weight += K @ (Y - X @ self.weight)
            
            # [QUAN TRỌNG] Dọn rác
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