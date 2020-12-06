from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from maf_layer import MAFLayer
from batch_norm_layer import BatchNormLayer


class MAF(nn.Module):
    def __init__(
        self, dim: int, n_layers: int, hidden_dims: List[int], use_reverse: bool = True
    ):
        """
        Args:
            dim: Dimension of input. E.g.: dim = 784 when using MNIST.
            n_layers: Number of layers in the MAF (= number of stacked MADEs).
            hidden_dims: List with of sizes of the hidden layers in each MADE. 
            use_reverse: Whether to reverse the input vector in each MADE. 
        """
        super().__init__()
        self.dim = dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(MAFLayer(dim, hidden_dims, reverse=use_reverse))
            self.layers.append(BatchNormLayer(dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0])
        # Forward pass.
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum += log_det

        return x, log_det_sum

    def backward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0])
        # Backward pass.
        for layer in reversed(self.layers):
            x, log_det = layer.backward(x)
            log_det_sum += log_det

        return x, log_det_sum
