import copy
import gc
import torch
from torch import nn
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
        
        # W hiện tại
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), **factory_kwargs))
        # W_ref (Snapshot)
        self.register_buffer("w_ref", torch.zeros((self.buffer_size, 0), **factory_kwargs))
        # R (Inverse Covariance) - Nằm trên GPU để tránh OOM RAM
        self.register_buffer("R", torch.eye(self.buffer_size, **factory_kwargs) / self.gamma)
        self.class_means = [] 
        self.class_vars = []
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0
        self.prev_known_class = 0 
        
    def update_fc(self, nb_classes):
        # Lưu Snapshot
        # if self.cur_task >= 0:
        #     self.w_ref = self.weight.clone().detach()
            
        self.cur_task += 1
        self.prev_known_class = self.known_class 
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

    def forward(self, x, new_forward=False):
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

    # @torch.no_grad()
    # def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
    #     try: from torch.amp import autocast
    #     except ImportError: from torch.cuda.amp import autocast
        
    #     with autocast('cuda', enabled=False):
    #         X = self.backbone(X).float() 
    #         X = self.buffer(X) 
    #         X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()
            
    #         # --- Expand Weight ---
    #         num_targets = Y.shape[1]
    #         if num_targets > self.weight.shape[1]:
    #             increment_size = num_targets - self.weight.shape[1]
    #             tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device)
    #             self.weight = torch.cat((self.weight, tail), dim=1)
                
    #             if self.w_ref.shape[1] > 0 and self.w_ref.shape[1] < num_targets:
    #                  ref_tail = torch.zeros((self.w_ref.shape[0], num_targets - self.w_ref.shape[1]), device=self.weight.device)
    #                  self.w_ref = torch.cat((self.w_ref, ref_tail), dim=1)

    #         elif num_targets < self.weight.shape[1]:
    #             increment_size = self.weight.shape[1] - num_targets
    #             tail = torch.zeros((Y.shape[0], increment_size), device=Y.device)
    #             Y = torch.cat((Y, tail), dim=1)
            
    #         # --- RLS Update (Standard - KHÔNG CÓ KÉO ở đây) ---
    #         # Xử lý R trực tiếp trên GPU
    #         term = torch.eye(X.shape[0], device=X.device) + X @ self.R @ X.T
    #         jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
    #         try: K = torch.linalg.solve(term + jitter, X @ self.R); K = K.T
    #         except: K = self.R @ X.T @ torch.inverse(term + jitter)
            
    #         self.R -= K @ X @ self.R
    #         self.weight += K @ (Y - X @ self.weight)
            
    #         # Đã loại bỏ hoàn toàn việc chuyển R về CPU -> Hết lỗi OOM
    #         del term, jitter, K, X, Y
    #         # if torch.cuda.is_available(): torch.cuda.empty_cache() # Có thể comment lại nếu muốn nhanh hơn
 

    #fit sinh mẫu giả
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Hàm fit cải tiến:
        1. Cập nhật thống kê Mean/Var cho class mới.
        2. Sinh dữ liệu giả cho class cũ (Pseudo-Replay).
        3. Tính RLS trên CPU theo từng Chunk nhỏ (tránh tràn RAM).
        """
        
        # 1. Chuyển dữ liệu thật về CPU ngay lập tức
        X_cpu_all = X.cpu().float()
        Y_cpu_all = Y.cpu().float()
        
        # Xóa tham chiếu GPU
        del X, Y
        
        # --- BƯỚC 1: CẬP NHẬT THỐNG KÊ (Class Mới) ---
        # Lưu ý: Code này giả định fit được gọi với batch đủ lớn hoặc re-fit toàn bộ data
        # để thống kê chính xác.
        if self.training:
            unique_classes = torch.argmax(Y_cpu_all, dim=1).unique()
            for c in unique_classes:
                c = c.item()
                # Mở rộng list nếu class mới xuất hiện
                if c >= len(self.class_means):
                    while len(self.class_means) <= c:
                        self.class_means.append(None)
                        self.class_vars.append(None)
                
                # Lấy feature của class c trong batch này
                mask = (torch.argmax(Y_cpu_all, dim=1) == c)
                features_c = X_cpu_all[mask]
                
                # Tính Mean/Var và lưu lại (CPU)
                # Dùng detach để chắc chắn không dính graph
                mean_c = features_c.mean(dim=0).detach()
                var_c = (features_c.var(dim=0, unbiased=False) + 1e-5).detach()
                
                self.class_means[c] = mean_c
                self.class_vars[c] = var_c

        # --- BƯỚC 2: SINH DATA GIẢ (Cho Class Cũ) ---
        if self.prev_known_class > 0:
            X_pseudo_list = []
            Y_pseudo_list = []
            
            # Số lượng mẫu giả mỗi class cũ
            # Để cân bằng, có thể lấy bằng số lượng trung bình class mới trong batch
            # Hoặc fix cứng con số này (ví dụ 20-50 mẫu/class)
            samples_per_class = 20 
            
            for c in range(self.prev_known_class):
                if c < len(self.class_means) and self.class_means[c] is not None:
                    mean = self.class_means[c] # Đã ở CPU
                    std = torch.sqrt(self.class_vars[c]) # Đã ở CPU
                    
                    # Sinh nhiễu Gaussian
                    noise = torch.randn(samples_per_class, self.buffer_size)
                    X_fake = noise * std + mean
                    
                    # Tạo nhãn one-hot
                    Y_fake = torch.zeros(samples_per_class, Y_cpu_all.shape[1])
                    Y_fake[:, c] = 1.0
                    
                    X_pseudo_list.append(X_fake)
                    Y_pseudo_list.append(Y_fake)
            
            if len(X_pseudo_list) > 0:
                X_pseudo = torch.cat(X_pseudo_list, dim=0)
                Y_pseudo = torch.cat(Y_pseudo_list, dim=0)
                
                # Gộp Thật + Giả
                X_cpu_all = torch.cat((X_cpu_all, X_pseudo), dim=0)
                Y_cpu_all = torch.cat((Y_cpu_all, Y_pseudo), dim=0)

        # --- BƯỚC 3: RLS CHUNKING (Chống Tràn RAM) ---
        BATCH_CHUNK = 256 # Kích thước an toàn cho RAM
        total_samples = X_cpu_all.shape[0]
        
        # Đảm bảo R ở CPU
        if self.R.device.type != 'cpu': self.R = self.R.cpu()
        
        # Kéo Weight về CPU để tính toán
        weight_cpu = self.weight.cpu()
        
        # Expand Weight nếu số class tăng lên
        num_targets = Y_cpu_all.shape[1]
        if num_targets > weight_cpu.shape[1]:
            tail = torch.zeros((weight_cpu.shape[0], num_targets - weight_cpu.shape[1]))
            weight_cpu = torch.cat((weight_cpu, tail), dim=1)
        elif num_targets < weight_cpu.shape[1]:
            tail = torch.zeros((Y_cpu_all.shape[0], weight_cpu.shape[1] - num_targets))
            Y_cpu_all = torch.cat((Y_cpu_all, tail), dim=1)

        # Vòng lặp tính toán từng miếng nhỏ
        for start_idx in range(0, total_samples, BATCH_CHUNK):
            end_idx = min(start_idx + BATCH_CHUNK, total_samples)
            
            X_batch = X_cpu_all[start_idx:end_idx]
            Y_batch = Y_cpu_all[start_idx:end_idx]
            
            # --- RLS Math Core ---
            # Chỉ tốn RAM cho ma trận [256, 256] và [256, 16384]
            term = torch.eye(X_batch.shape[0]) + X_batch @ self.R @ X_batch.T
            jitter = 1e-6 * torch.eye(term.shape[0])
            
            try: 
                K = torch.linalg.solve(term + jitter, X_batch @ self.R)
                K = K.T
            except: 
                K = self.R @ X_batch.T @ torch.inverse(term + jitter)
            
            # Update
            self.R -= K @ X_batch @ self.R
            weight_cpu += K @ (Y_batch - X_batch @ weight_cpu)
            
            # Dọn rác ngay lập tức
            del term, jitter, K, X_batch, Y_batch

        # Update xong, đẩy Weight mới lên GPU
        self.weight = weight_cpu.to(self.device)
        
        # Xóa biến to
        del X_cpu_all, Y_cpu_all, weight_cpu
        gc.collect() # Force dọn RAM
    # --- HÀM MỚI: Weight Merging (Gọi 1 lần cuối task) ---
    def weight_merging(self, alpha=0.2):
        if self.cur_task > 0 and self.w_ref.shape[1] > 0:
            print(f"--> [Weight Merging] Blending with alpha={alpha}...")
            old_cols = self.prev_known_class
            
            W_new = self.weight[:, :old_cols]
            W_ref = self.w_ref[:, :old_cols].to(self.weight.device)
            
            # W_final = (1-alpha) * W_new + alpha * W_ref
            self.weight[:, :old_cols] = (1 - alpha) * W_new + alpha * W_ref

    def extract_feature(self, x): return self.backbone(x)
    
    def collect_projections(self, mode='threshold', val=0.95):
        print(f"--> [IncNet] GPM Collect...")
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].compute_projection_matrix(mode, val)

    def apply_gpm_to_grads(self, scale=1.0):
        for j in range(self.backbone.layer_num): self.backbone.noise_maker[j].apply_gradient_projection(scale)


    def forward_normal_fc(self, x, new_forward=False):
        if new_forward: h = self.backbone(x, new_forward=True)
        else: h = self.backbone(x)
        h = self.buffer(h.to(self.buffer.weight.dtype))
        h = h.to(self.normal_fc.weight.dtype)
        return {"logits": self.normal_fc(h)['logits']}