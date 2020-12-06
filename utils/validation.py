import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal


def val_maf(model, train, val_loader):
    model.eval()
    val_loss = []
    _, _ = model.forward(train.float())
    for batch in val_loader:
        u, log_det = model.forward(batch.float())
        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        val_loss.extend(negloglik_loss.tolist())

    N = len(val_loader.dataset)
    loss = np.sum(val_loss) / N
    print(
        "Validation loss: {:.4f} +/- {:.4f}".format(
            loss, 2 * np.std(val_loss) / np.sqrt(N)
        )
    )
    return loss


def val_made(model, val_loader):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch in val_loader:
            out = model.forward(batch.float())
            mu, logp = torch.chunk(out, 2, dim=1)
            u = (batch - mu) * torch.exp(0.5 * logp)

            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)
            negloglik_loss = torch.mean(negloglik_loss)

            val_loss.append(negloglik_loss)

    N = len(val_loader)
    loss = np.sum(val_loss) / N
    print(
        "Validation loss: {:.4f} +/- {:.4f}".format(
            loss, 2 * np.std(val_loss) / np.sqrt(N)
        )
    )
    return loss
