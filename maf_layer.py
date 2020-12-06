from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from made import MADE


class MAFLayer(nn.Module):
    def __init__(self, dim: int, hidden_dims: List[int], reverse: bool):
        super(MAFLayer, self).__init__()
        self.dim = dim
        self.made = MADE(dim, hidden_dims, gaussian=True, seed=None)
        self.reverse = reverse

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.made(x.float())
        mu, logp = torch.chunk(out, 2, dim=1)
        u = (x - mu) * torch.exp(0.5 * logp)
        u = u.flip(dims=(1,)) if self.reverse else u
        log_det = 0.5 * torch.sum(logp, dim=1)
        return u, log_det

    def backward(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        u = u.flip(dims=(1,)) if self.reverse else u
        x = torch.zeros_like(u)
        for dim in range(self.dim):
            out = self.made(x)
            mu, logp = torch.chunk(out, 2, dim=1)
            mod_logp = torch.clamp(-0.5 * logp, max=10)
            x[:, dim] = mu[:, dim] + u[:, dim] * torch.exp(mod_logp[:, dim])
        log_det = torch.sum(mod_logp, axis=1)
        return x, log_det
