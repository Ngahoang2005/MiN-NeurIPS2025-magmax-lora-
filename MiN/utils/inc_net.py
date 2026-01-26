import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

# ============================================================
# Random Buffer (GIỮ NGUYÊN)
# ============================================================
class RandomBuffer(nn.Module):
    def __init__(self, in_features, buffer_size, device):
        super().__init__()
        self.in_features = in_features
        self.buffer_size = buffer_size
        self.device = device

        self.weight = nn.Parameter(
            torch.randn(in_features, buffer_size, device=device) * 0.02,
            requires_grad=False
        )

    def forward(self, x):
        return F.relu(x @ self.weight)


# ============================================================
# PiNoise – BiLoRA style, DISJOINT FREQUENCY
# ============================================================
class PiNoise(nn.Module):
    """
    1 PiNoise = 1 task
    Each task owns exactly k frequency bins
    """
    def __init__(self, in_dim, task_id, k, device):
        super().__init__()

        self.in_dim = in_dim
        self.task_id = task_id
        self.k = k
        self.device = device

        self.n_freq = in_dim // 2
        start = task_id * k
        end = start + k

        assert end <= self.n_freq, \
            f"[PiNoise] freq overflow: task {task_id}, end={end}, max={self.n_freq}"

        self.freq_slice = slice(start, end)

        print(f"[PiNoise] Task {task_id} freq {start}–{end-1}")

        # generators
        self.mu = nn.Sequential(
            nn.Linear(k, 128),
            nn.GELU(),
            nn.Linear(128, k)
        )

        self.sigma = nn.Sequential(
            nn.Linear(k, 128),
            nn.GELU(),
            nn.Linear(128, k),
            nn.Softplus()
        )

        # projection
        self.w_up = nn.Parameter(torch.randn(in_dim, in_dim) * 0.02)

        self.to(device)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def update_noise(self):
        pass  # MiN compatibility

    def forward(self, x):
        B, D = x.shape
        assert D == self.in_dim

        freq = torch.zeros(B, self.n_freq, device=x.device)
        eps = torch.randn(B, self.k, device=x.device)

        freq[:, self.freq_slice] = self.mu(eps) + eps * self.sigma(eps)

        full = torch.cat([freq, freq.flip(1)], dim=1)[:, :self.in_dim]
        noise = full @ self.w_up

        return noise


# ============================================================
# Backbone + PiNoise Stack
# ============================================================
class BackboneWithPiNoise(nn.Module):
    def __init__(self, backbone, k, device):
        super().__init__()
        self.backbone = backbone
        self.k = k
        self.device = device

        self.noise_maker = nn.ModuleList()
        self.cur_task = 0

        self.in_dim = backbone.out_dim

    def begin_task(self):
        # freeze old
        for n in self.noise_maker:
            n.freeze()

        # add new
        pn = PiNoise(
            in_dim=self.in_dim,
            task_id=self.cur_task,
            k=self.k,
            device=self.device
        )
        self.noise_maker.append(pn)
        self.cur_task += 1

    def forward(self, x, new_forward=True):
        feat = self.backbone(x)

        if new_forward and len(self.noise_maker) > 0:
            noise = 0.0
            for pn in self.noise_maker:
                noise = noise + pn(feat)
            feat = feat + noise

        return feat


# ============================================================
# MiN Base Network (GIỮ RLS + RandomBuffer)
# ============================================================
class MiNbaseNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.device = args["device"]
        self.k = args["k"]
        self.gamma = args["gamma"]
        self.buffer_size = args["buffer_size"]

        from backbones.pretrained_backbone import get_pretrained_backbone
        backbone = get_pretrained_backbone(args)

        self.backbone = BackboneWithPiNoise(
            backbone=backbone,
            k=self.k,
            device=self.device
        )

        self.feature_dim = backbone.out_dim

        # Random Buffer
        self.buffer = RandomBuffer(
            in_features=self.feature_dim,
            buffer_size=self.buffer_size,
            device=self.device
        )

        # RLS classifier
        self.register_buffer(
            "weight",
            torch.zeros(self.buffer_size, 0, device=self.device)
        )
        self.register_buffer(
            "R",
            torch.eye(self.buffer_size, device=self.device) / self.gamma
        )

        self.normal_fc = None
        self.known_class = 0

    # ======================================================
    # Task control
    # ======================================================
    def begin_task(self):
        self.backbone.begin_task()

    def after_task_magmax_merge(self):
        # normalize each task noise
        with torch.no_grad():
            for pn in self.backbone.noise_maker:
                w = pn.w_up
                pn.w_up.copy_(w / (w.norm() + 1e-6))

    # ======================================================
    # FC
    # ======================================================
    def update_fc(self, num_new_classes):
        self.known_class += num_new_classes

        new_fc = nn.Linear(self.buffer_size, self.known_class, bias=False).to(self.device)

        if self.normal_fc is not None:
            old = self.normal_fc.out_features
            new_fc.weight.data[:old] = self.normal_fc.weight.data

        self.normal_fc = new_fc

    # ======================================================
    # RLS fit (GIỮ)
    # ======================================================
    @torch.no_grad()
    def fit(self, x, y):
        feat = self.backbone(x)
        z = self.buffer(feat)

        if y.shape[1] > self.weight.shape[1]:
            pad = y.shape[1] - self.weight.shape[1]
            self.weight = torch.cat(
                [self.weight, torch.zeros(self.buffer_size, pad, device=self.device)],
                dim=1
            )

        I = torch.eye(z.shape[0], device=self.device)
        K = torch.inverse(I + z @ self.R @ z.T)
        self.R -= self.R @ z.T @ K @ z @ self.R
        self.weight += self.R @ z.T @ (y - z @ self.weight)

    # ======================================================
    # Forward
    # ======================================================
    def forward(self, x, new_forward=True):
        feat = self.backbone(x, new_forward=new_forward)
        z = self.buffer(feat)
        logits = z @ self.weight
        return {"logits": logits}

    def forward_normal_fc(self, x):
        feat = self.backbone(x, new_forward=True)
        z = self.buffer(feat)
        logits = self.normal_fc(z)
        return {"logits": logits}
