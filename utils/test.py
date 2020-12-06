import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal


def test_maf(model, train, test_loader):
    model.eval()
    test_loss = []
    _, _ = model.forward(train)
    with torch.no_grad():
        for batch in test_loader:
            u, log_det = model.forward(batch.float())

            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= log_det

            test_loss.extend(negloglik_loss)
    N = len(test_loss)
    print(
        "Test loss: {:.4f} +/- {:.4f}".format(
            np.mean(test_loss), 2 + np.std(test_loss) / np.sqrt(N)
        )
    )


def test_made(model, test_loader):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for batch in test_loader:
            out = model.forward(batch.float())
            mu, logp = torch.chunk(out, 2, dim=1)
            u = (batch - mu) * torch.exp(0.5 * logp)

            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

            test_loss.extend(negloglik_loss)

    N = len(test_loss)

    print(
        "Test loss: {:.4f} +/- {:.4f}".format(
            np.mean(test_loss), 2 + np.std(test_loss) / np.sqrt(N)
        )
    )
