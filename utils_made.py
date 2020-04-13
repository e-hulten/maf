import torch
from torch.nn import functional as F
from torch.distributions import Bernoulli, MultivariateNormal

import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import math


def train_one_epoch_made(model, epoch, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch in train_loader:
        out = model.forward(batch.float())
        mu, logp = torch.chunk(out, 2, dim=1)
        u = (batch - mu) * torch.exp(0.5 * logp)

        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

        negloglik_loss = torch.mean(negloglik_loss)

        negloglik_loss.backward()
        train_loss += negloglik_loss.item()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = np.sum(train_loss) / len(train_loader)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss


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
    print(
        "Validation loss: {:.4f} +/- {:.4f}".format(
            np.sum(val_loss) / N, 2 * np.std(val_loss) / np.sqrt(N)
        )
    )
    return np.sum(val_loss) / N


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
