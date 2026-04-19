"""
OTTA adapters — ported from the paper's reference implementation:
  src/eeg_continual/tta/alignment.py
  src/eeg_continual/tta/norm.py
  src/eeg_continual/tta/base.py

Adapted for EEGNetFea / DeepConvNetFea / ShallowConvNetFea:
  - model.forward(x) expects x shape (B, C, T),  internally does unsqueeze(1)
  - model.forward returns (logits, features) tuple
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg


# ──────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────
class TTAMethod(nn.Module):
    """Single-sample online TTA base class (mirrors paper tta/base.py)."""

    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.model = model
        self.config = config
        self.configure_model()

    def forward(self, x):
        assert x.shape[0] == 1, "OTTA requires batch_size=1 (single-sample)"
        return self.forward_and_adapt(x)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        raise NotImplementedError

    def configure_model(self):
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# Euclidean Alignment  (mirrors paper tta/alignment.py exactly)
# ──────────────────────────────────────────────────────────────
class Alignment(TTAMethod):
    """
    Online Euclidean Alignment.
    x input shape: (1, C, T)  — 3D, same as model expects.
    Internally computes covmat as (C,C), updates running reference,
    whitens x, then passes to model.
    """

    def __init__(self, model: nn.Module, config: dict, reference: np.ndarray = None):
        self.reference = reference
        self.counter = 0
        super().__init__(model, config)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        x_aligned = self.align_data(x, self.config.get("alignment"))
        logits, _ = self.model(x_aligned)
        return logits

    def align_data(self, x, alignment):
        """
        x: torch.Tensor shape (1, C, T)
        covmat = x @ x.T / T  shape (C, C)
        running reference updated with weighted average (same as paper).
        """
        # (1, C, T) → covmat (C, C)
        covmat = torch.matmul(x, x.transpose(1, 2)).detach().cpu().numpy()[0]
        self.counter += 1

        if self.reference is not None:
            weights = [1 - 1 / self.counter, 1 / self.counter]
            if alignment == "euclidean":
                self.reference = np.average(
                    [self.reference, covmat], axis=0, weights=weights
                )
            else:
                raise NotImplementedError(f"Alignment '{alignment}' not implemented")
        else:
            self.reference = covmat

        R_op = linalg.inv(linalg.sqrtm(self.reference))
        R_tensor = torch.tensor(R_op, dtype=torch.float32, device=x.device)
        # x: (1, C, T) → matmul with (C, C) → (1, C, T)
        x_aligned = torch.matmul(R_tensor, x)
        return x_aligned

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)


# ──────────────────────────────────────────────────────────────
# RobustBN  (mirrors paper tta/norm.py → RobustBN)
# ──────────────────────────────────────────────────────────────
class RobustBN(nn.Module):
    """
    Drop-in replacement for nn.BatchNorm2d that blends frozen source
    statistics with incoming batch statistics using fixed momentum alpha.

    train mode:  mean = (1-alpha)*source_mean + alpha*batch_mean  (updates source)
    eval  mode:  uses stored source_mean / source_var directly
    """

    @staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d):
                replace_mods.append((parent, name, RobustBN(child, alpha)))
            else:
                replace_mods.extend(RobustBN.find_bns(child, alpha))
        return replace_mods

    @staticmethod
    def adapt_model(model, alpha=0.001):
        """Replace all BatchNorm2d in model with RobustBN in-place."""
        replace_mods = RobustBN.find_bns(model, alpha)
        print(f"[RobustBN] Replacing {len(replace_mods)} BatchNorm2d layers (alpha={alpha})")
        for parent, name, module in replace_mods:
            setattr(parent, name, module)
        return model

    def __init__(self, bn_layer: nn.BatchNorm2d, momentum: float):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum   # alpha in the paper
        self.eps = bn_layer.eps

        if (bn_layer.track_running_stats
                and bn_layer.running_mean is not None
                and bn_layer.running_var is not None):
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var",  deepcopy(bn_layer.running_var))

        self.weight = deepcopy(bn_layer.weight)
        self.bias   = deepcopy(bn_layer.bias)

    def forward(self, x):
        # x: (B, C, H, W)  — standard 4D BN input from Conv2d output
        if self.training:
            b_var, b_mean = torch.var_mean(
                x, dim=[0, 2, 3], unbiased=False, keepdim=False
            )
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var  = (1 - self.momentum) * self.source_var  + self.momentum * b_var
            # Update source statistics for the next sample
            self.source_mean = deepcopy(mean.detach())
            self.source_var  = deepcopy(var.detach())
        else:
            mean = self.source_mean
            var  = self.source_var

        x = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


# ──────────────────────────────────────────────────────────────
# Norm  (mirrors paper tta/norm.py → Norm)
# ──────────────────────────────────────────────────────────────
class Norm(TTAMethod):
    """
    RobustBN-based OTTA, with optional EA pre-processing.
    configure_model() replaces all BN layers with RobustBN and sets
    model to train mode so statistics are updated on every sample.

    forward returns only logits (features tuple stripped internally).
    """

    def __init__(self, model: nn.Module, config: dict, reference: np.ndarray = None):
        # EA state (only used when config['alignment'] is truthy)
        self.reference = reference
        self.counter   = 0
        super().__init__(model, config)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        # Optional EA (reuses Alignment.align_data, self carries reference/counter)
        if self.config.get("alignment", False):
            x = Alignment.align_data(self, x, self.config.get("alignment"))
        logits, _ = self.model(x)
        return logits

    def configure_model(self):
        # Replace BN layers with RobustBN (source stats frozen from trained weights)
        self.model = RobustBN.adapt_model(
            self.model, alpha=self.config.get("alpha", 0.001)
        )
        self.model.requires_grad_(False)
        # train mode: RobustBN.forward() updates source_mean/source_var each call
        self.model.train()