import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
# [FIX]: Import từ torch.amp thay vì torch.cuda.amp để hỗ trợ cú pháp autocast('cuda', ...)
from torch.amp import autocast 

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
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        self.register_buffer("weight", torch.zeros((self.buffer_size, 0), **factory_kwargs))

        # [DPCR STORAGE]
        self._compressed_stats = [] 
        self._mu_list = []          
        self._class_counts = []     

        self.temp_phi = {}
        self.temp_mu = {}
        self.temp_count = {}

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

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
            if new_fc.bias is not None: nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

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
            if hasattr(self.backbone.blocks[j], 'norm1'):
                for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            if hasattr(self.backbone.blocks[j], 'norm2'):
                for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters(): p.requires_grad = True

    def collect_projections(self, mode='threshold', val=0.95):
        print(f"--> [IncNet] Collecting Projections (Mode: {mode}, Val: {val})...")
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].compute_projection_matrix(mode=mode, val=val)

    def apply_gpm_to_grads(self, scale=1.0):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].apply_gradient_projection(scale=scale)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.eval()
        # autocast('cuda', ...) chỉ hoạt động đúng nếu import từ torch.amp
        with autocast('cuda', enabled=False):
            feat = self.backbone(X).float()
            feat = self.buffer(feat)

        labels = torch.argmax(Y, dim=1)
        for i in range(feat.shape[0]):
            label = labels[i].item()
            f = feat[i:i+1]
            
            if label not in self.temp_phi:
                self.temp_phi[label] = torch.zeros(self.buffer_size, self.buffer_size, device=self.device)
                self.temp_mu[label] = torch.zeros(self.buffer_size, device=self.device)
                self.temp_count[label] = 0
            
            self.temp_phi[label] += f.t() @ f
            self.temp_mu[label] += f.squeeze(0)
            self.temp_count[label] += 1

    @torch.no_grad()
    def solve_temporary_analytic(self):
        device = self.device
        total_phi = torch.zeros(self.buffer_size, self.buffer_size, device=device)
        total_q = torch.zeros(self.buffer_size, self.known_class, device=device)
        
        for c in range(len(self._compressed_stats)):
            V, S = self._compressed_stats[c]
            V, S = V.to(device), S.to(device)
            phi_c = (V @ torch.diag(S)) @ V.t()
            total_phi += phi_c
            total_q[:, c] = self._mu_list[c].to(device) * self._class_counts[c]
            del V, S, phi_c

        for label in self.temp_phi:
            total_phi += self.temp_phi[label]
            total_q[:, label] = self.temp_mu[label]

        reg_phi = total_phi + self.gamma * torch.eye(self.buffer_size, device=device)
        try:
            W = torch.linalg.solve(reg_phi, total_q)
        except:
            W = torch.inverse(reg_phi) @ total_q
            
        self.normal_fc.weight.data = W.t().to(self.normal_fc.weight.dtype)
        self.temp_phi, self.temp_mu, self.temp_count = {}, {}, {}

    @torch.no_grad()
    def finalize_task_stats(self):
        device = self.device
        COMPRESS_RANK = 256 
        
        for label in sorted(self.temp_phi.keys()):
            raw_phi = self.temp_phi[label]
            try:
                S, V = torch.linalg.eigh(raw_phi) 
            except:
                U, S, _ = torch.svd(raw_phi)
                V = U

            if S.shape[0] > COMPRESS_RANK:
                S_top = S[-COMPRESS_RANK:]
                V_top = V[:, -COMPRESS_RANK:]
            else:
                S_top = S
                V_top = V

            self._compressed_stats.append((V_top.cpu(), S_top.cpu()))
            self._mu_list.append((self.temp_mu[label] / self.temp_count[label]).cpu())
            self._class_counts.append(self.temp_count[label])
        
        self.temp_phi, self.temp_mu, self.temp_count = {}, {}, {}

    @torch.no_grad()
    def reconstruct_classifier_dpcr(self, P_drift=None, known_classes_boundary=0):
        device = self.device
        num_total_classes = len(self._compressed_stats)
        
        if num_total_classes > self.weight.shape[1]:
            new_cols = num_total_classes - self.weight.shape[1]
            tail = torch.zeros((self.buffer_size, new_cols), device=device)
            new_weight = torch.cat((self.weight, tail), dim=1)
            del self.weight
            self.register_buffer("weight", new_weight)

        total_phi = torch.zeros(self.buffer_size, self.buffer_size, device=device)
        total_q = torch.zeros(self.buffer_size, num_total_classes, device=device)

        for c in range(num_total_classes):
            V, S = self._compressed_stats[c]
            V, S = V.to(device), S.to(device)
            phi_c = (V @ torch.diag(S)) @ V.t()
            mu_c = self._mu_list[c].to(device)

            if P_drift is not None and c < known_classes_boundary:
                P_cs = V @ V.t()
                P_dual = P_drift @ P_cs
                phi_c = P_dual.t() @ phi_c @ P_dual
                mu_c = mu_c @ P_dual
                
                try:
                    S_new, V_new = torch.linalg.eigh(phi_c)
                    S_top = S_new[-256:]
                    V_top = V_new[:, -256:]
                    self._compressed_stats[c] = (V_top.cpu(), S_top.cpu())
                    self._mu_list[c] = mu_c.cpu()
                except:
                    pass 

            total_phi += phi_c
            total_q[:, c] = mu_c * self._class_counts[c]
            del V, S, phi_c, mu_c

        reg_phi = total_phi + self.gamma * torch.eye(self.buffer_size, device=device)
        try:
            new_w = torch.linalg.solve(reg_phi, total_q)
        except:
            new_w = torch.inverse(reg_phi) @ total_q
            
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