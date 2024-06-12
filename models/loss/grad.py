from __future__ import annotations
from typing import Any, Literal
import torch
from torch import Tensor
import torch.nn as nn


class Gradient(nn.Module):
    def __init__(self, penalty: Literal["l1", "l2"] = "l2") -> None:
        super().__init__()
        self.penalty = penalty
        return

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        return super().__call__(*args, **kwds)

    def forward(self, flow: Tensor):
        dim_num = len(flow.shape) - 2
        nabla = []
        for dim in range(2, 2 + dim_num):
            index = torch._dim_arange(flow, dim)
            # derivative
            ddim = flow.index_select(dim, index[:-1]) - flow.index_select(dim, index[1:])
            # norm
            if self.penalty == "l2":
                ddim = ddim ** 2
            else:
                ddim = ddim.abs()
            # mean
            nabla.append(ddim.mean())
        loss = torch.stack(nabla).mean()
        return loss
