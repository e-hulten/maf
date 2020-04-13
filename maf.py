import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from made import MADE
from batchnorm import BatchNorm


class MAF(nn.Module):
    """
    dim: dimension of data input, e.g.: dim = 784 when using MNIST
    n_maf: number of MADEs in the flow
    hidden_dims: list with the size of the hidden layers in each MADE, e.g.: hidden_dims = [1000,1000]
    use_reverse:
    """

    def __init__(self, dim, n_maf, hidden_dims, use_reverse=True):
        super().__init__()
        self.dim = dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        for _ in range(n_maf):
            self.layers.append(MAFLayer(dim, hidden_dims, reverse=use_reverse))
            self.layers.append(BatchNorm(dim))

    def forward(self, x):
        log_det_sum = torch.zeros(x.shape[0])

        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum += log_det

        return x, log_det_sum

    def backward(self, x):
        log_det_sum = torch.zeros(x.shape[0])

        for maf in reversed(self.layers):
            x, log_det = maf.backward(x)
            log_det_sum += log_det

        return x, log_det_sum


class MAFLayer(nn.Module):
    def __init__(self, dim, hidden_dims, reverse):
        super(MAFLayer, self).__init__()
        self.dim = dim
        self.made = MADE(dim, hidden_dims, gaussian=True, seed=None)
        self.reverse = reverse

    def forward(self, x):
        out = self.made(x.float())
        mu, logp = torch.chunk(out, 2, dim=1)
        u = (x - mu) * torch.exp(0.5 * logp)
        u = u.flip(dims=(1,)) if self.reverse else u
        log_det = 0.5 * torch.sum(logp, dim=1)
        return u, log_det

    def backward(self, u):
        u = u.flip(dims=(1,)) if self.reverse else u
        x = torch.zeros_like(u)
        for dim in range(self.dim):
            out = self.made(x)
            mu, logp = torch.chunk(out, 2, dim=1)
            mod_logp = torch.clamp(-0.5 * logp, max=10)
            x[:, dim] = mu[:, dim] + u[:, dim] * torch.exp(mod_logp[:, dim])
        log_det = torch.sum(mod_logp, axis=1)
        return x, log_det

