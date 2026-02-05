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
        # Dùng float32 để tránh lỗi singular matrix khi tính nghịch đảo
        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) # Trọng số của Analytic Classifier

        self.register_buffer("R", torch.eye(self.buffer_size, **factory_kwargs) / self.gamma)

        # --- LẮP THÊM DPCR STORAGE ---
        self._compressed_stats = {} # Lưu (V, S) sau khi nén
        self._mu_list = {}          # Lưu mean vector per class
        self._class_counts = {}     # Lưu số lượng sample per class
        self.temp_phi = {}          # Lưu Phi tạm thời trước khi nén
        self.temp_mu = {}
        self.temp_count = {}
        # Normal FC: Dùng để train Gradient Descent cho Noise Generator
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

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
    @torch.no_grad()
    def accumulate_stats(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.eval()
        # Đảm bảo input khớp thiết bị và kiểu dữ liệu (FP32 cho thống kê)
        ref_p = next(self.backbone.parameters())
        X = X.to(device=self.device, dtype=ref_p.dtype)
        
        # Trích xuất đặc trưng và đưa về FP32 để tính toán ma trận hiệp phương sai
        feat = self.buffer(self.backbone(X)).float()
        labels = torch.argmax(Y, dim=1)
        
        for i in range(feat.shape[0]):
            label = labels[i].item()
            f = feat[i:i+1]
            if label not in self.temp_phi:
                self.temp_phi[label] = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
                self.temp_mu[label] = torch.zeros(self.buffer_size, device=self.device)
                self.temp_count[label] = 0
            self.temp_phi[label] += (f.t() @ f).detach()
            self.temp_mu[label] += f.squeeze(0).detach()
            self.temp_count[label] += 1
    @torch.no_grad()
    def compress_stats(self):
        RANK = 256 # Rank nén của DPCR
        torch.cuda.empty_cache()
        for label in sorted(list(self.temp_phi.keys())):
            # Phân tích trị riêng để nén
            S, V = torch.linalg.eigh(self.temp_phi[label])
            S_top, V_top = (S[-RANK:], V[:, -RANK:]) if S.shape[0] > RANK else (S, V)
            
            # Lưu vào bộ nhớ nén (có thể đẩy sang CPU nếu GPU quá đầy, ở đây tao để GPU)
            self._compressed_stats[label] = (V_top.detach(), S_top.detach())
            self._mu_list[label] = (self.temp_mu[label] / self.temp_count[label]).detach()
            self._class_counts[label] = self.temp_count[label]
            
            # Xóa dữ liệu thô
            del self.temp_phi[label], self.temp_mu[label]
        self.temp_phi, self.temp_mu, self.temp_count = {}, {}, {}
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        [ORIGINAL RLS]: Hàm gốc 100%. 
        """
        try:
            from torch.amp import autocast
        except ImportError:
            from torch.cuda.amp import autocast

        with autocast('cuda', enabled=False):
            # [FIX]: Đặt tên biến rõ ràng để tránh lỗi del
            X_input = X.to(self.device).float()
            
            # Feature Extraction -> Đặt tên là feat
            feat = self.backbone(X_input).float()
            feat = self.buffer(feat) 
            
            Y = Y.to(self.weight.device).float()

            # Tự động mở rộng chiều
            num_targets = Y.shape[1]
            if num_targets > self.weight.shape[1]:
                increment_size = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.buffer_size, increment_size), device=self.weight.device)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.weight.shape[1]:
                increment_size = self.weight.shape[1] - num_targets
                tail = torch.zeros((Y.shape[0], increment_size), device=Y.device)
                Y = torch.cat((Y, tail), dim=1)

            # RLS Update Batch-wise
            # Sử dụng feat thay vì X
            term = torch.eye(feat.shape[0], device=feat.device) + feat @ self.R @ feat.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            try:
                K = torch.linalg.solve(term + jitter, feat @ self.R)
                K = K.T
            except:
                K = self.R @ feat.T @ torch.inverse(term + jitter)

            self.R -= K @ feat @ self.R
            self.weight += K @ (Y - feat @ self.weight)
            
            # [FIX]: Xóa đúng biến đã khai báo
            del term, jitter, K, X_input, Y, feat
            torch.cuda.empty_cache()
    
    @torch.no_grad()
    def solve_analytic(self, P_drift=None, boundary=0, init_mode=False):
        # Tính toán số lượng class hiện có
        all_keys = list(self._compressed_stats.keys()) + list(self.temp_phi.keys())
        num_total = max(all_keys) + 1 if all_keys else 0
        
        # Mở rộng ma trận trọng số
        if num_total > self.weight.shape[1]:
            new_w = torch.zeros((self.buffer_size, num_total), device=self.device)
            new_w[:, :self.weight.shape[1]] = self.weight
            self.register_buffer("weight", new_w)

        total_phi = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
        total_q = torch.zeros(self.buffer_size, num_total, device=self.device)

        # Hồi phục từ dữ liệu nén
        for c, (V, S) in self._compressed_stats.items():
            phi_c = (V @ torch.diag(S)) @ V.t()
            mu_c = self._mu_list[c]
            
            # Áp dụng Drift Correction cho các class cũ (Task < hiện tại)
            if P_drift is not None and c < boundary:
                P_drift = P_drift.to(self.device)
                P_cs = V @ V.t()
                phi_c = P_drift.t() @ P_cs.t() @ phi_c @ P_cs @ P_drift
                mu_c = mu_c @ P_cs @ P_drift
                
            total_phi += phi_c
            total_q[:, c] = mu_c * self._class_counts[c]

        # Giải hệ phương trình tuyến tính
        reg = self.gamma * torch.eye(self.buffer_size, device=self.device)
        W = torch.linalg.solve(total_phi + reg, total_q)
        
        if init_mode:
            self.normal_fc.weight.data[:W.shape[1]] = W.t().to(self.normal_fc.weight.dtype)
        else:
            self.weight.data = F.normalize(W, p=2, dim=0)
    # =========================================================================
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