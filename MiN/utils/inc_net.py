# import copy
# import logging
# import math
# import numpy as np
# import torch
# from torch import nn, Tensor
# from torch.utils.data import DataLoader
# from backbones.pretrained_backbone import get_pretrained_backbone
# from backbones.linears import SimpleLinear, SplitCosineLinear, CosineLinear
# from einops.layers.torch import Rearrange, Reduce
# from einops import rearrange, reduce, repeat
# from torch.nn import functional as F
# import scipy.stats as stats
# import timm
# import random
# from torch.cuda.amp import autocast 

# class BaseIncNet(nn.Module):
#     # ... (Giữ nguyên phần đầu) ...
#     def __init__(self, args: dict):
#         super(BaseIncNet, self).__init__()
#         self.args = args
#         self.backbone = get_pretrained_backbone(args)
#         self.feature_dim = self.backbone.out_dim
#         self.fc = None

#     def update_fc(self, nb_classes):
#         fc = self.generate_fc(self.feature_dim, nb_classes)
#         if self.fc is not None:
#             nb_output = self.fc.out_features
#             weight = copy.deepcopy(self.fc.weight.data)
#             bias = copy.deepcopy(self.fc.bias.data)
#             fc.weight.data[:nb_output] = weight
#             fc.bias.data[:nb_output] = bias
#         del self.fc
#         self.fc = fc
#         start = self.known_class
#         end = self.known_class + nb_classes
#         if not hasattr(self, 'task_class_indices'):
#             self.task_class_indices = []
#         self.task_class_indices.append(list(range(start, end)))
#     @staticmethod
#     def generate_fc(in_dim, out_dim):
#         fc = SimpleLinear(in_dim, out_dim)
#         return fc

#     def forward(self, x):
#         hyper_features = self.backbone(x)
#         logits = self.fc(hyper_features)['logits']
#         return {'features': hyper_features, 'logits': logits}

# class RandomBuffer(torch.nn.Linear):
#     def __init__(self, in_features: int, buffer_size: int, device):
#         super(torch.nn.Linear, self).__init__()
#         self.bias = None
#         self.in_features = in_features
#         self.out_features = buffer_size
#         factory_kwargs = {"device": device, "dtype": torch.float32}
#         self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
#         self.register_buffer("weight", self.W)
#         self.reset_parameters()

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         X = X.to(self.weight.dtype)
#         return F.relu(X @ self.W)

# class MiNbaseNet(nn.Module):
#     def __init__(self, args: dict):
#         super(MiNbaseNet, self).__init__()
#         self.args = args
#         self.backbone = get_pretrained_backbone(args)
#         self.device = args['device']
#         self.gamma = args['gamma']
#         self.buffer_size = args['buffer_size']
#         self.feature_dim = self.backbone.out_dim
#         self.task_prototypes = []

#         self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)
#         factory_kwargs = {"device": self.device, "dtype": torch.float32}

#         weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
#         self.register_buffer("weight", weight)
#         self.register_buffer("R_global", torch.eye(self.buffer_size, **factory_kwargs) / self.gamma)
#         self.R_list = nn.ParameterList()
#         self.task_weights = nn.ParameterList() 
#         self.task_class_indices = [] # Để lưu vết class của từng task
        

      

#         self.Pinoise_list = nn.ModuleList()
#         self.normal_fc = None
#         self.cur_task = -1
#         self.known_class = 0
#         self.fc2 = nn.ModuleList()
        
#         # [QUAN TRỌNG] Thêm H, Hy để tránh lỗi AttributeError trong Trainer
#         self.register_buffer("H", torch.zeros((self.buffer_size, self.buffer_size), **factory_kwargs))
#         self.register_buffer("Hy", torch.zeros((self.buffer_size, 0), **factory_kwargs))

#     def set_grad_checkpointing(self, enable=True):
#         if hasattr(self.backbone, 'set_grad_checkpointing'):
#             self.backbone.set_grad_checkpointing(enable)
#         elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'set_grad_checkpointing'):
#              self.backbone.model.set_grad_checkpointing(enable)
#         elif hasattr(self.backbone, 'grad_checkpointing'):
#             self.backbone.grad_checkpointing = enable

#     def forward_fc(self, features):
#         features = features.to(self.weight)
#         return features @ self.weight

#     @property
#     def in_features(self) -> int:
#         return self.weight.shape[0]

#     @property
#     def out_features(self) -> int:
#         return self.weight.shape[1]

#     def update_fc(self, nb_classes):
#         self.cur_task += 1
#         start = self.known_class
#         self.known_class += nb_classes
#         self.task_class_indices.append(list(range(start, self.known_class)))

#         # 1. Khởi tạo ma trận R riêng biệt cho Task hiện tại
#         # R_t = (X^T X + \gamma I)^{-1}
#         new_R = nn.Parameter(torch.eye(self.buffer_size, device=self.device) / self.gamma, requires_grad=False)
#         self.R_list.append(new_R)

#         # 2. Khởi tạo W riêng biệt cho Expert hiện tại
#         new_w = nn.Parameter(torch.zeros((self.buffer_size, nb_classes), device=self.device))
#         self.task_weights.append(new_w)

#         # 3. [FIX ERROR] Khởi tạo normal_fc ngay tại đây để tránh lỗi NoneType
#         self.normal_fc = SimpleLinear(self.buffer_size, self.known_class, bias=(self.cur_task==0))
#     @torch.no_grad()
#     def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
#         # RLS cực kỳ nhạy cảm với độ chính xác, tắt autocast để tính toán trên float32
#         with torch.amp.autocast('cuda', enabled=False):
#             # 1. Trích xuất đặc trưng (Backbone đang ở mode Trainer set)
#             X = self.backbone(X).float() 
#             X = self.buffer(X) # [Batch, Buffer_Size]
#             X, Y = X.to(self.device), Y.to(self.device).float()
            
#             batch_size = X.shape[0]
#             num_classes_total = Y.shape[1]
#             I = torch.eye(batch_size, device=self.device)

#             # --- PHẦN 1: CẬP NHẬT GLOBAL WEIGHTS (Cho nhánh Universal) ---
#             # Tự động mở rộng Global W nếu số class tăng lên
#             if num_classes_total > self.weight.shape[1]:
#                 diff = num_classes_total - self.weight.shape[1]
#                 tail = torch.zeros((self.buffer_size, diff), device=self.device)
#                 self.weight = torch.cat((self.weight, tail), dim=1)

#             # RLS cho Global
#             # term = I + X @ R @ X.T
#             term_glob = I + X @ self.R_global @ X.T
#             term_glob.diagonal().add_(1e-6) # Ridge-like regularization
#             K_inv_glob = torch.inverse(term_glob)
#             Gain_glob = self.R_global @ X.T @ K_inv_glob
            
#             # Cập nhật R_global và W_global
#             self.R_global -= Gain_glob @ (self.R_global @ X.T).T
#             self.weight += Gain_glob @ (Y - X @ self.weight)

#             # --- PHẦN 2: CẬP NHẬT EXPERT RIÊNG BIỆT (R riêng + W riêng) ---
#             # Lấy đúng R và W của Expert hiện tại
#             R_curr = self.R_list[self.cur_task]
#             W_curr = self.task_weights[self.cur_task]
            
#             # Chỉ lấy nhãn (Y) thuộc về các class của task hiện tại
#             curr_class_indices = self.task_class_indices[self.cur_task]
#             Y_task = Y[:, curr_class_indices] # [Batch, Num_Task_Classes]

#             # RLS cho Expert (Dùng R riêng của task đó)
#             term_spec = I + X @ R_curr @ X.T
#             term_spec.diagonal().add_(1e-6)
#             K_inv_spec = torch.inverse(term_spec)
#             Gain_spec = R_curr @ X.T @ K_inv_spec
            
#             # Cập nhật dữ liệu vào Parameter (Dùng .data để bypass autograd)
#             R_curr.data -= Gain_spec @ (R_curr @ X.T).T
#             W_curr.data += Gain_spec @ (Y_task - X @ W_curr)

#         # Giải phóng bộ nhớ tạm
#         del X, Y, I, Gain_glob, Gain_spec, K_inv_glob, K_inv_spec
    
#     def forward(self, x, new_forward: bool = False):
#         if new_forward:
#             hyper_features = self.backbone(x, new_forward=True)
#         else:
#             hyper_features = self.backbone(x)
        
#         hyper_features = hyper_features.to(self.weight.dtype)
#         logits = self.forward_fc(self.buffer(hyper_features))
#         return {'logits': logits}
    
#     def update_task_prototype(self, prototype):
#         if isinstance(prototype, torch.Tensor):
#             self.task_prototypes[-1] = prototype.detach().cpu()
#         else:
#             self.task_prototypes[-1] = prototype

#     def extend_task_prototype(self, prototype):
#         if isinstance(prototype, torch.Tensor):
#             self.task_prototypes.append(prototype.detach().cpu())
#         else:
#             self.task_prototypes.append(prototype)

#     def extract_feature(self, x):
#         hyper_features = self.backbone(x)
#         return hyper_features

#     def forward_normal_fc(self, x, new_forward: bool = False):
#         if new_forward:
#             hyper_features = self.backbone(x, new_forward=True)
#         else:
#             hyper_features = self.backbone(x)
#         hyper_features = self.buffer(hyper_features)
#         hyper_features = hyper_features.to(self.normal_fc.weight.dtype) 
#         logits = self.normal_fc(hyper_features)['logits']
#         return {"logits": logits}

#     def update_noise(self):
#         # Lấy Mean của các class proto để init noise (đơn giản hóa thành 1 vector/task)
#         task_means = []
#         if len(self.task_prototypes) > 0:
#             for p in self.task_prototypes:
#                 # p là [Num_Class, Dim], mean lại thành [Dim]
#                 task_means.append(p.mean(dim=0))
        
#         for j in range(self.backbone.layer_num):
#             self.backbone.noise_maker[j].update_noise()
#             self.backbone.noise_maker[j].init_weight_noise(task_means)

#     def unfreeze_noise(self):
#         for j in range(self.backbone.layer_num):
#             self.backbone.noise_maker[j].unfreeze_noise()

#     def init_unfreeze(self):
#         for j in range(self.backbone.layer_num):
#             for param in self.backbone.noise_maker[j].parameters():
#                 param.requires_grad = True
#             for p in self.backbone.blocks[j].norm1.parameters():
#                 p.requires_grad = True
#             for p in self.backbone.blocks[j].norm2.parameters():
#                 p.requires_grad = True
#         for p in self.backbone.norm.parameters():
#             p.requires_grad = True

#     def set_noise_mode(self, mode):
#         if hasattr(self.backbone, 'noise_maker'):
#             for m in self.backbone.noise_maker:
#                 m.active_task_idx = mode

#     def forward_tuna_combined(self, x, targets=None, top_k=3, tau=0.1):
#         self.eval()
#         batch_size = x.shape[0]
#         num_tasks = len(self.task_prototypes)
        
#         # BƯỚC 1: UNIVERSAL LOGITS (Nền tảng tri thức chung)
#         self.set_noise_mode(-2)
#         with torch.no_grad():
#             feat_uni = self.backbone(x)
#             logits_uni = self.forward_fc(self.buffer(feat_uni))

#         if num_tasks == 0: return {'logits': logits_uni}

#         # BƯỚC 2: ROUTING MAHALANOBIS (Top-K)
#         self.set_noise_mode(-3)
#         with torch.no_grad():
#             feat_clean = self.backbone(x)
#             X = self.buffer(feat_clean) 
            
#             task_scores = []
#             for t_idx in range(num_tasks):
#                 R_t = self.R_list[t_idx] 
#                 Mu = self.task_prototypes[t_idx].to(self.device) 
                
#                 # dist^2 = xRx' + muRmu' - 2xRmu'
#                 xRx = torch.sum((X @ R_t) * X, dim=-1)
#                 muRmu = torch.sum((Mu @ R_t) * Mu, dim=-1)
#                 xRmu = X @ R_t @ Mu.T
#                 dist_sq = xRx.unsqueeze(1) + muRmu.unsqueeze(0) - 2 * xRmu
                
#                 min_dist_task, _ = dist_sq.min(dim=1)
#                 task_scores.append(-min_dist_task) # Score cao = gần
            
#             all_scores = torch.stack(task_scores, dim=1) # [Batch, Num_Tasks]
            
#             # Tính Soft Trọng số cho tất cả Expert (giống Eq. 12 trong MIN)
#             # Dùng tau nhỏ để tập trung vào các Expert đứng đầu
#             routing_weights = F.softmax(all_scores / tau, dim=1) 
            
#             # Lấy Top-K Expert có trọng số cao nhất
#             top_weights, top_task_ids = torch.topk(routing_weights, k=min(top_k, num_tasks), dim=1)

#         # BƯỚC 3: SOFT ENSEMBLE EXPERTS
#         # Thay vì gán cứng, ta cộng dồn logit có trọng số
#         combined_expert_logits = torch.zeros_like(logits_uni)
        
#         with torch.no_grad():
#             # Duyệt qua các Expert xuất hiện trong Top-K của cả batch
#             unique_top_tasks = top_task_ids.unique()
#             for t in unique_top_tasks:
#                 t_idx = t.item()
#                 # Tìm xem ảnh nào trong batch có Expert t_idx nằm trong Top-K
#                 batch_mask = (top_task_ids == t_idx).any(dim=1)
                
#                 if not batch_mask.any(): continue
                
#                 self.set_noise_mode(t_idx)
#                 feat_spec = self.backbone(x[batch_mask])
#                 raw_logits_task = self.buffer(feat_spec) @ self.task_weights[t_idx]
                
#                 # Lấy trọng số tương ứng của Expert t_idx cho từng ảnh
#                 # Chỉ những ảnh có t_idx trong top_k mới lấy weight, còn lại = 0
#                 w_t = torch.zeros((batch_size, 1), device=self.device)
#                 for k in range(top_weights.shape[1]):
#                     k_mask = (top_task_ids[:, k] == t_idx)
#                     w_t[k_mask] = top_weights[k_mask, k].unsqueeze(1)
                
#                 # Masking & Weighted Add
#                 idx = self.task_class_indices[t_idx]
#                 expert_contribution = torch.zeros_like(logits_uni[batch_mask])
#                 expert_contribution[:, idx] = raw_logits_task
                
#                 combined_expert_logits[batch_mask] += w_t[batch_mask] * expert_contribution

#         # BƯỚC 4: ENSEMBLE FINAL
#         # MIN gốc sử dụng Noise để mask, ở đây ta ensemble Logits
#         final_logits = logits_uni + combined_expert_logits
        
#         # Tính routing_acc cho expert top-1 để debug
#         best_task_ids = top_task_ids[:, 0]
#         routing_acc = -1.0
#         if targets is not None:
#             true_task_ids = torch.zeros_like(best_task_ids)
#             for t_idx, indices in enumerate(self.task_class_indices):
#                 mask_t = (targets >= indices[0]) & (targets <= indices[-1])
#                 true_task_ids[mask_t] = t_idx
#             routing_acc = (best_task_ids == true_task_ids).float().mean().item() * 100

#         self.set_noise_mode(-2)
#         return {'logits': final_logits, 'routing_acc': routing_acc}






















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
        self.register_buffer("R_global", torch.eye(self.buffer_size, **factory_kwargs) / self.gamma)
        self.R_list = nn.ParameterList()
        self.task_weights = nn.ParameterList() 
        self.task_class_indices = [] # Để lưu vết class của từng task
        

      

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
        start = self.known_class
        self.known_class += nb_classes
        self.task_class_indices.append(list(range(start, self.known_class)))

        # 1. Khởi tạo ma trận R riêng biệt cho Task hiện tại
        # R_t = (X^T X + \gamma I)^{-1}
        new_R = nn.Parameter(torch.eye(self.buffer_size, device=self.device) / self.gamma, requires_grad=False)
        self.R_list.append(new_R)

        # 2. Khởi tạo W riêng biệt cho Expert hiện tại
        new_w = nn.Parameter(torch.zeros((self.buffer_size, nb_classes), device=self.device))
        self.task_weights.append(new_w)

        # 3. [FIX ERROR] Khởi tạo normal_fc ngay tại đây để tránh lỗi NoneType
        self.normal_fc = SimpleLinear(self.buffer_size, self.known_class, bias=(self.cur_task==0))
    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        # RLS cực kỳ nhạy cảm với độ chính xác, tắt autocast để tính toán trên float32
        with torch.amp.autocast('cuda', enabled=False):
            # 1. Trích xuất đặc trưng (Backbone đang ở mode Trainer set)
            X = self.backbone(X).float() 
            X = self.buffer(X) # [Batch, Buffer_Size]
            X, Y = X.to(self.device), Y.to(self.device).float()
            
            batch_size = X.shape[0]
            num_classes_total = Y.shape[1]
            I = torch.eye(batch_size, device=self.device)

            # --- PHẦN 1: CẬP NHẬT GLOBAL WEIGHTS (Cho nhánh Universal) ---
            # Tự động mở rộng Global W nếu số class tăng lên
            if num_classes_total > self.weight.shape[1]:
                diff = num_classes_total - self.weight.shape[1]
                tail = torch.zeros((self.buffer_size, diff), device=self.device)
                self.weight = torch.cat((self.weight, tail), dim=1)

            # RLS cho Global
            # term = I + X @ R @ X.T
            term_glob = I + X @ self.R_global @ X.T
            term_glob.diagonal().add_(1e-6) # Ridge-like regularization
            K_inv_glob = torch.inverse(term_glob)
            Gain_glob = self.R_global @ X.T @ K_inv_glob
            
            # Cập nhật R_global và W_global
            self.R_global -= Gain_glob @ (self.R_global @ X.T).T
            self.weight += Gain_glob @ (Y - X @ self.weight)

            # --- PHẦN 2: CẬP NHẬT EXPERT RIÊNG BIỆT (R riêng + W riêng) ---
            # Lấy đúng R và W của Expert hiện tại
            R_curr = self.R_list[self.cur_task]
            W_curr = self.task_weights[self.cur_task]
            
            # Chỉ lấy nhãn (Y) thuộc về các class của task hiện tại
            curr_class_indices = self.task_class_indices[self.cur_task]
            Y_task = Y[:, curr_class_indices] # [Batch, Num_Task_Classes]

            # RLS cho Expert (Dùng R riêng của task đó)
            term_spec = I + X @ R_curr @ X.T
            term_spec.diagonal().add_(1e-6)
            K_inv_spec = torch.inverse(term_spec)
            Gain_spec = R_curr @ X.T @ K_inv_spec
            
            # Cập nhật dữ liệu vào Parameter (Dùng .data để bypass autograd)
            R_curr.data -= Gain_spec @ (R_curr @ X.T).T
            W_curr.data += Gain_spec @ (Y_task - X @ W_curr)

        # Giải phóng bộ nhớ tạm
        del X, Y, I, Gain_glob, Gain_spec, K_inv_glob, K_inv_spec
    
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

    def forward_tuna_combined(self, x, targets=None, top_k=3, tau=0.1):
        self.eval()
        batch_size = x.shape[0]
        num_tasks = len(self.task_prototypes)
        
        # BƯỚC 1: LẤY LOGIT UNIVERSAL (Mode -2: Mixture) [cite: 31, 186]
        self.set_noise_mode(-2)
        with torch.no_grad():
            feat_uni = self.backbone(x)
            feat_buf_uni = self.buffer(feat_uni)
            # Giữ nguyên đặc trưng thô để nhân với FC chung [cite: 77]
            logits_uni = self.forward_fc(feat_buf_uni)

        if num_tasks == 0: return {'logits': logits_uni}

        # BƯỚC 2: ROUTING QUA N LẦN FORWARD 
        task_scores = []
        all_feat_specs = [] # Lưu lại đặc trưng thô của từng Expert

        with torch.no_grad():
            for t_idx in range(num_tasks):
                # A. Chuyển sang thấu kính của Expert t [cite: 113, 115]
                self.set_noise_mode(t_idx)
                
                # B. Forward để lấy đặc trưng qua Noise Maker t
                feat_t = self.backbone(x)
                feat_buf_t = self.buffer(feat_t)
                all_feat_specs.append(feat_buf_t) # Lưu đặc trưng thô
                
                # C. Tính Mahalanobis (Dùng đặc trưng thô, không Normalize)
                R_t = self.R_list[t_idx] 
                Mu = self.task_prototypes[t_idx].to(self.device) 
                
                # dist^2 = xRx' + muRmu' - 2xRmu'
                xRx = torch.sum((feat_buf_t @ R_t) * feat_buf_t, dim=-1)
                muRmu = torch.sum((Mu @ R_t) * Mu, dim=-1)
                xRmu = feat_buf_t @ R_t @ Mu.T
                
                dist_sq = xRx.unsqueeze(1) + muRmu.unsqueeze(0) - 2 * xRmu
                
                # Chọn khoảng cách nhỏ nhất trong số các class của task t
                min_dist_task, _ = dist_sq.min(dim=1)
                task_scores.append(-min_dist_task)

            # D. Tính trọng số Routing dựa trên Top-K [cite: 35, 102, 219]
            all_scores = torch.stack(task_scores, dim=1)
            routing_weights = F.softmax(all_scores / tau, dim=1) 
            top_weights, top_task_ids = torch.topk(routing_weights, k=min(top_k, num_tasks), dim=1)

        # BƯỚC 3: SOFT ENSEMBLE LOGITS (Sử dụng đặc trưng thô đã lưu) [cite: 104, 254]
        combined_expert_logits = torch.zeros_like(logits_uni)
        
        with torch.no_grad():
            # Duyệt qua các vị trí trong Top-K
            for k in range(top_weights.shape[1]):
                t_indices = top_task_ids[:, k] # Task ID được chọn ở rank k
                weights_k = top_weights[:, k].unsqueeze(1) # Trọng số tương ứng
                
                for t_idx in range(num_tasks):
                    mask = (t_indices == t_idx)
                    if not mask.any(): continue
                    
                    # Lấy đặc trưng thô đã tính từ thấu kính t tương ứng
                    feat_ready = all_feat_specs[t_idx][mask]
                    # Nhân trực tiếp với bộ trọng số Expert t [cite: 114]
                    raw_logits_task = feat_ready @ self.task_weights[t_idx]
                    
                    # Đưa vào đúng vị trí lớp trong classifier toàn cục
                    idx = self.task_class_indices[t_idx]
                    expert_contribution = torch.zeros_like(logits_uni[mask])
                    expert_contribution[:, idx] = raw_logits_task
                    
                    # Cộng dồn có trọng số vào kết quả Expert Ensemble
                    combined_expert_logits[mask] += weights_k[mask] * expert_contribution

        # BƯỚC 4: KẾT HỢP CUỐI CÙNG (Universal + Experts) [cite: 31, 104, 254]
        final_logits = logits_uni + combined_expert_logits
        
        # --- TÍNH TỶ LỆ CHỌN ĐÚNG EXPERT ---
        routing_acc = -1.0
        if targets is not None:
            best_task_ids = top_task_ids[:, 0]
            true_task_ids = torch.zeros_like(best_task_ids)
            for t_idx, indices in enumerate(self.task_class_indices):
                mask_t = (targets >= indices[0]) & (targets <= indices[-1])
                true_task_ids[mask_t] = t_idx
            routing_acc = (best_task_ids == true_task_ids).float().mean().item() * 100

        self.set_noise_mode(-2)
        return {'logits': final_logits, 'routing_acc': routing_acc}