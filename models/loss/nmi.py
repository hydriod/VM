from __future__ import annotations
from typing import Any
from math import sqrt, pi
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from utils import EPS


class GaussianHistogram(nn.Module):
    def __init__(
            self,
            bins: int = 256,
            min: float | None = -0.25,
            max: float | None = 1.25,
            xsigma: float | None = 1 / sqrt(2 * pi),
    ):
        super().__init__()
        self.bins = bins
        self.min = min if min is not None else 0.
        self.max = max if max is not None else 1.
        self.xsigma = xsigma if xsigma is not None else 1 / sqrt(2 * pi)

    @property
    def delta(self):
        return (self.max - self.min) / self.bins

    @property
    def sigma(self):
        return self.delta * self.xsigma

    @property
    def centers(self):
        return self.min + self.delta * (torch.arange(self.bins, dtype=torch.float) + 0.5)

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        """
        Parameters
        ---
        x1: Tensor
        x2: Tensor
        mask: Tensor

        Returns
        ---
        histogram: Tensor
            Size([b, bins, bins])
        """
        return super().__call__(*args, **kwds)

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor | None = None):
        b, *_ = x1.shape
        centers = self.centers.to(x1.device).expand(b, -1)
        mask = mask if mask is not None else torch.ones_like(x1)
        # Brightness Distance [batch, x1center_idx, x2center_idx, pixel_idx]
        x1 = x1.reshape(b, 1, 1, -1) - centers.reshape(b, -1, 1, 1)
        x2 = x2.reshape(b, 1, 1, -1) - centers.reshape(b, 1, -1, 1)
        mask = mask.reshape(b, 1, 1, -1)
        x = x1**2 + x2**2
        # Kernel function
        x = 1. / (sqrt(2 * pi) * self.sigma) * torch.exp(-0.5 * x / self.sigma**2) * self.delta
        # x1 = torch.sigmoid((x1 + self.delta/2) / self.sigma) - torch.sigmoid((x1 - self.delta/2) / self.sigma)
        # Create Histogram
        x = x * mask
        x = x.sum(dim=-1)
        return x


class NMI(nn.Module):
    def __init__(
        self,
        bins=64,
        xsigma: float | None = 1 / sqrt(2 * pi),
        down_scale: int = 2,
    ) -> None:
        super().__init__()
        self.down_scale = down_scale
        self.histc = GaussianHistogram(bins=bins, xsigma=xsigma)

    def _lamp(self, tensor: Tensor):
        """
        Normalize [0, 1]
        """
        b, *chw = tensor.shape
        tensor = tensor.reshape(b, -1)
        tensor = tensor - tensor.min(dim=1, keepdim=True)[0]
        tensor = tensor / tensor.max(dim=1, keepdim=True)[0]
        tensor = tensor.reshape(b, *chw)
        return tensor

    def mi(self, m: Tensor, f: Tensor, mask: Tensor | None = None):
        """バッチごとの相互情報量"""
        assert m.shape == f.shape, f"m.shape({m.shape}) != f.shape({f.shape})"
        b, c, *hwd = m.shape
        hwd = tuple(map(lambda x: x//self.down_scale, hwd))
        mode = "bilinear" if len(m.shape) == 4 else "trilinear"
        m = F.interpolate(m, hwd, mode=mode)
        f = F.interpolate(f, hwd, mode=mode)

        m = self._lamp(m)
        f = self._lamp(f)

        hist = self.histc(m, f, mask)
        p_mf = hist / (hist.sum([1, 2], keepdim=True) + EPS)
        # p(m) * p(f)
        p_m_f = p_mf.sum(2, keepdim=True) * p_mf.sum(1, keepdim=True)

        I_mf = torch.sum(p_mf * torch.log2(p_mf / (p_m_f + EPS) + EPS), dim=[1, 2])
        # torch._assert(not I_mf.isnan().any(), "NAN")
        return I_mf, p_mf

    def nmi(self, m: Tensor, f: Tensor, mask: Tensor | None = None):
        mutual_info, hist = self.mi(m, f, mask)
        mutual_info_max = self.mi(f, f, mask)[0]
        return mutual_info / (mutual_info_max + EPS), hist

    def forward(self, m, f, mask=None):
        normalized_MI = self.nmi(m, f, mask)[0]
        return 1. - normalized_MI.mean()

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        """
        Parameters
        ---
        m: Tensor
        f: Tensor
        mask: Optional[Tensor]
        """
        return super().__call__(*args, **kwds)
