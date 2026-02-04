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

class RandomBuffer(torch.nn.Linear):
    """Lớp Random Projection để mở rộng chiều đặc trưng lên 16384"""
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
        return F.relu(X @ self.weight)

class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        self.gamma = args['gamma'] # Hệ số điều chuẩn (Gamma = 100)
        self.buffer_size = args['buffer_size'] # 16384
        self.feature_dim = self.backbone.out_dim 

        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        # Weight Analytic (sẽ được tái cấu trúc liên tục)
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), **factory_kwargs))

        # [DPCR STORAGE]
        self._phi_list = []      # List chứa ma trận Covariance (CPU)
        self._mu_list = []       # List chứa Prototype (CPU)
        self._class_counts = []  # List chứa số lượng mẫu

        # Biến tạm để gom thống kê
        self.temp_phi = {}
        self.temp_mu = {}
        self.temp_count = {}

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        
        # Tạo Normal FC (Dùng cho SGD training Noise)
        if self.cur_task > 0:
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
        if self.normal_fc is not None:
            old_nb_output = self.normal_fc.out_features
            with torch.no_grad():
                # Copy trọng số cũ sang
                new_fc.weight[:old_nb_output] = self.normal_fc.weight.data
                nn.init.constant_(new_fc.weight[old_nb_output:], 0.)
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None: nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    # --- Phần quản lý Noise ---
    def update_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    def after_task_magmax_merge(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].after_task_training()

    def unfreeze_noise(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_incremental()

    def init_unfreeze(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_task_0()

    # --- Phần Analytic Learning & DPCR ---

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Gom thống kê thô từ dữ liệu (thường là Augment)."""
        self.eval()
        with autocast('cuda', enabled=False):
            feat = self.backbone(X).float()
            feat = self.buffer(feat) # [Batch, 16384]

        labels = torch.argmax(Y, dim=1)
        for i in range(feat.shape[0]):
            label = labels[i].item()
            f = feat[i:i+1]
            
            if label not in self.temp_phi:
                # Tính trên GPU cho nhanh
                self.temp_phi[label] = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
                self.temp_mu[label] = torch.zeros(self.buffer_size, device=self.device)
                self.temp_count[label] = 0
            
            self.temp_phi[label] += f.t() @ f
            self.temp_mu[label] += f.squeeze(0)
            self.temp_count[label] += 1

    @torch.no_grad()
    def solve_temporary_analytic(self):
        """
        Dùng cho bước 'Init Fit'. Giải RLS tạm thời để lấy trọng số tốt gán cho normal_fc.
        Sau đó XÓA stats tạm vì sắp train noise làm drift feature.
        """
        device = self.device
        total_phi = torch.zeros(self.buffer_size, self.buffer_size, device=device)
        total_q = torch.zeros(self.buffer_size, self.known_class, device=device)
        
        # 1. Load stats cũ (từ CPU lên GPU)
        for c in range(len(self._phi_list)):
            total_phi += self._phi_list[c].to(device)
            total_q[:, c] = self._mu_list[c].to(device) * self._class_counts[c]
        
        # 2. Cộng stats mới (đang ở temp GPU)
        for label in self.temp_phi:
            total_phi += self.temp_phi[label]
            total_q[:, label] = self.temp_mu[label] # Đã nhân N lúc tích lũy

        # 3. Giải RLS: W = (Phi + Gamma*I)^-1 @ Q
        reg_phi = total_phi + self.gamma * torch.eye(self.buffer_size, device=device)
        try:
            W = torch.linalg.solve(reg_phi, total_q)
        except:
            W = torch.inverse(reg_phi) @ total_q
            
        # 4. Gán vào normal_fc (Transpose vì Linear lưu [Out, In])
        self.normal_fc.weight.data = W.t().to(self.normal_fc.weight.dtype)
        print("--> [Init] Synced Analytic Weights to Normal FC.")

        # 5. Quan trọng: Reset temp stats
        self.temp_phi, self.temp_mu, self.temp_count = {}, {}, {}

    @torch.no_grad()
    def finalize_task_stats(self):
        """
        Dùng cho bước 'Final Fit'. Lưu stats vào bộ nhớ chính thức (CPU) để DPCR dùng sau.
        """
        for label in sorted(self.temp_phi.keys()):
            # Đẩy về CPU để tránh OOM (16k*16k float32 ~ 1GB/class)
            self._phi_list.append(self.temp_phi[label].cpu())
            self._mu_list.append((self.temp_mu[label] / self.temp_count[label]).cpu())
            self._class_counts.append(self.temp_count[label])
        
        self.temp_phi, self.temp_mu, self.temp_count = {}, {}, {}

    @torch.no_grad()
    def reconstruct_classifier_dpcr(self, P_drift=None, known_classes_boundary=0):
        """
        DPCR: Hiệu chỉnh Drift và Tái cấu trúc Classifier.
        P_drift: Ma trận TSSP [16k, 16k].
        known_classes_boundary: Số lượng class cũ cần hiệu chỉnh.
        """
        device = self.device
        num_total_classes = len(self._phi_list)
        
        # Mở rộng buffer weight cho đủ số class
        if num_total_classes > self.weight.shape[1]:
            new_cols = num_total_classes - self.weight.shape[1]
            tail = torch.zeros((self.buffer_size, new_cols), device=device)
            new_weight = torch.cat((self.weight, tail), dim=1)
            del self.weight
            self.register_buffer("weight", new_weight)

        total_phi = torch.zeros(self.buffer_size, self.buffer_size, device=device)
        total_q = torch.zeros(self.buffer_size, num_total_classes, device=device)

        # Duyệt qua từng class để cộng dồn và hiệu chỉnh
        for c in range(num_total_classes):
            # Load từng class lên GPU
            phi_c = self._phi_list[c].to(device)
            mu_c = self._mu_list[c].to(device)

            # --- DPCR CALIBRATION ---
            if P_drift is not None and c < known_classes_boundary:
                # 1. CSSP (Row-space projection)
                # Dùng SVD để tìm không gian con của class
                try:
                    U, S, _ = torch.linalg.svd(phi_c)
                except:
                    U, S, _ = torch.svd(phi_c)
                
                # Lấy các chiều quan trọng (Eigenvalue > epsilon)
                mask = S > 1e-5
                if mask.sum() > 0:
                    U_c = U[:, mask]
                    P_cs = U_c @ U_c.t()
                else:
                    P_cs = torch.eye(self.buffer_size, device=device)

                # 2. Dual-Projection: P = P_task @ P_class
                P_dual = P_drift @ P_cs
                
                # 3. Nắn chỉnh (Phi_new = P^T * Phi * P)
                phi_c = P_dual.t() @ phi_c @ P_dual
                mu_c = mu_c @ P_dual
                
                # Lưu lại ký ức đã nắn về CPU cho task sau
                self._phi_list[c] = phi_c.cpu()
                self._mu_list[c] = mu_c.cpu()

            # Cộng dồn vào tổng thể
            total_phi += phi_c
            total_q[:, c] = mu_c * self._class_counts[c]
            
            # Giải phóng bộ nhớ ngay lập tức
            del phi_c, mu_c

        # Giải Ridge Regression
        reg_phi = total_phi + self.gamma * torch.eye(self.buffer_size, device=device)
        try:
            new_w = torch.linalg.solve(reg_phi, total_q)
        except:
            new_w = torch.inverse(reg_phi) @ total_q
            
        # Category-wise Normalization (CN)
        self.weight.data = F.normalize(new_w, p=2, dim=0)

    def forward_fc(self, features):
        return features.to(self.weight.dtype) @ self.weight

    def forward(self, x, new_forward=False):
        h = self.backbone(x, new_forward=new_forward)
        logits = self.forward_fc(self.buffer(h))
        return {'logits': logits}

    def forward_normal_fc(self, x, new_forward=False):
        h = self.buffer(self.backbone(x, new_forward=new_forward).to(self.buffer.weight.dtype))
        return {"logits": self.normal_fc(h.to(self.normal_fc.weight.dtype))['logits']}

    def extract_feature(self, x): return self.backbone(x)