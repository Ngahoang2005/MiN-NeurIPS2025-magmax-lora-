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
        return X @ self.W


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
        # Dùng float32 để tránh lỗi singular matrix khi tính nghịch đảo
        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) # Trọng số của Analytic Classifier

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) # Ma trận hiệp phương sai đảo (Inverse Covariance Matrix)

        # Normal FC: Dùng để train Gradient Descent cho Noise Generator
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        self._saved_mean = {}
        self._saved_cov = {}
        self._saved_count = {}

    def update_fc(self, nb_classes):
        """
        Cập nhật lớp Normal FC (cho việc training Noise).
        Lớp Analytic FC (self.weight) sẽ tự động mở rộng trong hàm fit().
        """
        self.cur_task += 1
        self.known_class += nb_classes
        
        # Tạo mới Normal FC cho task hiện tại
        if self.cur_task > 0:
            # Task sau: Không dùng Bias để tránh bias vào lớp mới quá nhiều
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            # Task đầu: Có bias
            # [FIX LỖI TẠI ĐÂY]: Đổi fc thành new_fc
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
        if self.normal_fc is not None:
            # Sequential Init: Copy trọng số cũ
            old_nb_output = self.normal_fc.out_features
            with torch.no_grad():
                # Copy phần cũ
                new_fc.weight[:old_nb_output] = self.normal_fc.weight.data
                # Init phần mới về 0
                nn.init.constant_(new_fc.weight[old_nb_output:], 0.)
            
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            # Task đầu tiên
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None:
                nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    # =========================================================================
    # [MAGMAX & NOISE CONTROL SECTION]
    # =========================================================================
    
    def update_noise(self):
        """
        Gọi khi bắt đầu Task mới.
        Kích hoạt chế độ Sequential Initialization trong PiNoise.
        """
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    def after_task_magmax_merge(self):
        """
        Gọi sau khi kết thúc Task.
        Kích hoạt việc LƯU (Save) và TRỘN (Merge) tham số theo MagMax.
        """
        print(f"--> [IncNet] Task {self.cur_task}: Triggering Parameter-wise MagMax Merging...")
        for j in range(self.backbone.layer_num):
            # Hàm này nằm trong PiNoise
            self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        """Gọi cho Task > 0: Chỉ unfreeze Noise thưa"""
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_incremental()

    def init_unfreeze(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_task_0()
            
            # Giữ LayerNorm trainable ở Task 0 để ổn định base
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
        """Forward qua Analytic Classifier"""
        # Đảm bảo features cùng kiểu với trọng số RLS (float32)
        features = features.to(self.weight.dtype) 
        return features @ self.weight
# Trong class MiNbaseNet
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Phiên bản RLS tối ưu bộ nhớ (Memory-Efficient RLS)
        """
        # Tắt Autocast để tính toán chính xác FP32 (tránh lỗi Singular Matrix)
        try:
            from torch.amp import autocast
        except ImportError:
            from torch.cuda.amp import autocast

        with autocast('cuda', enabled=False):
            # 1. Feature Extraction & Expansion
            X = self.backbone(X).float() 
            X = self.buffer(X) 
            
            # Đảm bảo cùng device
            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            # 2. Mở rộng chiều của classifier nếu có lớp mới
            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.weight.shape[1]:
                # Trường hợp hiếm: Padding Y cho khớp weight cũ
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size), device=Y.device)
                Y = torch.cat((Y, tail), dim=1)

            # 3. RLS Update (Tối ưu OOM)
            # Công thức: P = (I + X R X^T)^-1
            # term kích thước [Batch x Batch]. Nếu Batch lớn (Buffer), cái này rất nặng.
            
            term = torch.eye(X.shape[0], device=X.device) + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            # Dùng linalg.solve nhanh và ổn định hơn torch.inverse
            try:
                # K = (X R X^T + I)^-1 @ (X R)
                # Kích thước [Batch x Buffer Dim]
                K = torch.linalg.solve(term + jitter, X @ self.R)
                K = K.T # Transpose về [Buffer Dim x Batch]
            except:
                # Fallback nếu lỗi
                K = self.R @ X.T @ torch.inverse(term + jitter)

            # Cập nhật R và Weight
            self.R -= K @ X @ self.R
            self.weight += K @ (Y - X @ self.weight)
            
            # [QUAN TRỌNG] Xóa ngay lập tức để giải phóng VRAM cho batch sau
            del term, jitter, K, X, Y    # =========================================================================
    # [FORWARD PASSES]
    # =========================================================================

    def forward(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # [SỬA]: Đảm bảo đặc trưng đồng nhất kiểu dữ liệu trước khi vào Buffer
        hyper_features = hyper_features.to(self.weight.dtype)
        
        # Buffer trả về ReLU(X @ W), forward_fc thực hiện X @ Weight
        logits = self.forward_fc(self.buffer(hyper_features))
        
        return {'logits': logits}
    def extract_feature(self, x):
        """Chỉ trích xuất đặc trưng từ Backbone"""
        return self.backbone(x)

    def forward_normal_fc(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # [SỬA]: Buffer thường chứa trọng số FP32, ép hyper_features lên FP32 
        # để phép nhân trong Buffer diễn ra chính xác trước khi đưa vào Classifier
        hyper_features = self.buffer(hyper_features.to(self.buffer.weight.dtype))
        
        # Sau đó ép về kiểu của normal_fc (thường là Half nếu dùng autocast)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}
    def collect_projections(self, mode='threshold', val=0.95):
        """
        Duyệt qua các lớp PiNoise và tính toán ma trận chiếu.
        """
        print(f"--> [IncNet] Collecting Projections (Mode: {mode}, Val: {val})...")
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].compute_projection_matrix(mode=mode, val=val)
    def apply_gpm_to_grads(self, scale=1.0):
        """
        Thực hiện chiếu trực giao gradient cho mu và sigma.
        """
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].apply_gradient_projection(scale=scale)
    # =========================================================================
    # [DPCR CORE FUNCTIONS ADDED HERE]
    # =========================================================================
    
    @torch.no_grad()
    def update_backbone_stats(self, inputs, targets):
        """
        Cập nhật Mean và Covariance Matrix theo phương pháp ổn định (Welford-like).
        Lưu ý: self._saved_cov bây giờ sẽ lưu 'M2' (Tổng bình phương sai số), 
        chưa chia cho N để tránh sai số tích lũy.
        """
        features = self.backbone(inputs).detach()
        
        for c in torch.unique(targets):
            c = c.item()
            mask = (targets == c)
            feats_batch = features[mask] # Batch dữ liệu mới (B)
            
            n_batch = feats_batch.shape[0]
            if n_batch == 0: continue

            # Tính thống kê của batch hiện tại
            mean_batch = feats_batch.mean(dim=0)
            # Tính Unbiased Covariance của batch * (N-1) hoặc Biased * N
            # Ở đây ta cần tổng sai số bình phương: sum((x - mu_batch)(x - mu_batch)^T)
            centered_batch = feats_batch - mean_batch
            m2_batch = centered_batch.T @ centered_batch # [Dim x Dim]

            if c not in self._saved_count:
                # Lần đầu tiên gặp class này
                self._saved_count[c] = n_batch
                self._saved_mean[c] = mean_batch
                self._saved_cov[c] = m2_batch # Lưu tổng M2, chưa chia
            else:
                # Cập nhật online (Stable Batched Welford Algorithm)
                n_old = self._saved_count[c]
                mean_old = self._saved_mean[c]
                m2_old = self._saved_cov[c]
                
                n_new = n_old + n_batch
                
                # 1. Cập nhật Mean
                # delta = mean_batch - mean_old
                # mean_new = mean_old + delta * n_batch / n_new
                delta = mean_batch - mean_old
                mean_new = mean_old + delta * (n_batch / n_new)
                
                # 2. Cập nhật Covariance Sum (M2)
                # M2_new = M2_old + M2_batch + delta^2 * (n_old * n_batch / n_new)
                m2_new = m2_old + m2_batch + torch.outer(delta, delta) * (n_old * n_batch / n_new)
                
                # Lưu lại
                self._saved_count[c] = n_new
                self._saved_mean[c] = mean_new
                self._saved_cov[c] = m2_new
    @torch.no_grad()
    def solve_dpcr(self, P_drift, boundary):
        """
        Phiên bản Robust: Chống lỗi NaN khi Covariance Matrix mất tính PSD.
        """
        HTH_old = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
        HTY_old = torch.zeros(self.buffer_size, boundary, device=self.device)
        
        P_drift = P_drift.to(self.device).float()
        
        print(f"--> [DPCR] Replaying stats with Robust Sampling (SVD Fallback)...")
        
        for c in range(boundary):
            mean_old = self._saved_mean[c]
            count = self._saved_count[c]
            
            # 1. Lấy Covariance thực (Unbiased)
            if count > 1:
                cov_real = self._saved_cov[c] / (count - 1)
            else:
                cov_real = torch.eye(self.feature_dim, device=self.device) * 1e-4
            
            # 2. TSSP: Transform Stats
            mean_new = mean_old @ P_drift
            cov_new = P_drift.t() @ cov_real @ P_drift
            
            # [FIX QUAN TRỌNG]: Ép đối xứng ma trận để giảm sai số
            cov_new = 0.5 * (cov_new + cov_new.t())
            
            # 3. CIP: Robust Sampling (Thủ công thay vì dùng MultivariateNormal)
            num_samples = min(count, 200)
            
            # Thêm Jitter để ổn định (Noise injection)
            jitter = 1e-4 * torch.eye(self.feature_dim, device=self.device)
            cov_final = cov_new + jitter
            
            try:
                # Cách 1: Thử Cholesky (Nhanh nhưng dễ lỗi)
                L = torch.linalg.cholesky(cov_final)
            except RuntimeError:
                # Cách 2: SVD / Eigendecomposition (Chậm hơn nhưng an toàn tuyệt đối)
                # print(f"    [Warning] Class {c}: Cholesky failed. Fallback to SVD.")
                e, v = torch.linalg.eigh(cov_final)
                # Cắt bỏ eigenvalue âm (Nguyên nhân gây NaN)
                e = torch.clamp(e, min=1e-6)
                # Tái tạo lại L sao cho L @ L.T = Cov
                L = v @ torch.diag(torch.sqrt(e))

            # Reparameterization Trick: Z = mu + epsilon * L
            # epsilon ~ N(0, I)
            epsilon = torch.randn(num_samples, self.feature_dim, device=self.device)
            sampled_z = mean_new.unsqueeze(0) + epsilon @ L.t()
            
            # 4. Qua Buffer & Tích lũy (Giữ nguyên)
            features_proj = self.buffer(sampled_z) # [N, Buffer_Dim]
            
            HTH_old += features_proj.t() @ features_proj
            
            y_onehot = torch.zeros(features_proj.shape[0], boundary, device=self.device)
            y_onehot[:, c] = 1.0
            HTY_old += features_proj.t() @ y_onehot
            
        return HTH_old, HTY_old
    def simple_ridge_solve(self, HTH, HTY, lambda_reg=0.1):
        """Giải Ridge Regression: W = (HTH + lambda*I)^-1 * HTY"""
        I = torch.eye(HTH.shape[0], device=self.device)
        try:
            W = torch.linalg.solve(HTH + lambda_reg * I, HTY)
        except:
            W = torch.inverse(HTH + lambda_reg * I) @ HTY
        return W

    def category_normalization(self, W):
        """CN: Chuẩn hóa norm của weight vector từng class"""
        norms = torch.norm(W, p=2, dim=0, keepdim=True)
        # Tránh chia cho 0
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        return W / norms