import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class MaskedSum(nn.Linear):
    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out)

    def initialise_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
        self, n_in, hidden_dims, random_order=False, seed=None, gaussian=False
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian is True else n_in
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.seed = seed
        self.gaussian = gaussian
        self.relu = torch.nn.ReLU(inplace=True)
        self.layers = []

        self.layers.append(MaskedSum(n_in, self.hidden_dims[0]))
        self.layers.append(self.relu)
        # hidden -> hidden
        for l in range(1, len(hidden_dims)):
            self.layers.append(MaskedSum(hidden_dims[l - 1], hidden_dims[l]))
            self.layers.append(self.relu)
        # hidden -> output
        self.layers.append(MaskedSum(hidden_dims[-1], self.n_out))

        # create model
        self.model = nn.Sequential(*self.layers)
        # get masks for the masked activations
        self.create_masks()

    def forward(self, x):
        return self.model(x) if self.gaussian else torch.sigmoid(self.model(x))

    def create_masks(self):
        np.random.seed(self.seed)
        self.masks = {}
        L = len(self.hidden_dims)  # number of hidden layers
        D = self.n_in  # number of inputs

        # if false, use the natural ordering [1,2,...,D]
        self.masks[0] = np.random.permutation(D) if self.random_order else np.arange(D)

        # set the connectivity number m for the hidden layers
        # m ~ DiscreteUniform[min_{prev_layer}(m), D-1]
        for l in range(L):
            self.masks[l + 1] = np.random.randint(
                self.masks[l].min(), D - 1, size=self.hidden_dims[l]
            )

        self.mask_matrix = []
        # create mask matrix for input->hidden_1->...->hidden_L
        # (i.e., excluding hidden_L->output)
        for mask_num in range(len(self.masks) - 1):
            m = self.masks[mask_num]  # current layer
            m_next = self.masks[mask_num + 1]  # next layer
            M = torch.zeros(len(m_next), len(m))  # mask matrix
            for i in range(len(m_next)):
                M[i, :] = torch.from_numpy((m_next[i] >= m).astype(int))
            self.mask_matrix.append(M)

        # create mask matrix for hidden_L->output
        m = self.masks[L]
        m_out = self.masks[0]
        M_out = torch.zeros(len(m_out), len(m))
        for i in range(len(m_out)):
            M_out[i, :] = torch.from_numpy((m_out[i] > m).astype(int))

        M_out = torch.cat((M_out, M_out), dim=0) if self.gaussian is True else M_out
        self.mask_matrix.append(M_out)

        # get masks for the layers of self.model
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedSum):
                module.initialise_mask(next(mask_iter))
