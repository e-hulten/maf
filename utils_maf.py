import torch
from torch.nn import functional as F
from torch.distributions import Bernoulli, MultivariateNormal

import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import math


def train_one_epoch_maf(model, epoch, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch in train_loader:
        u, log_det = model.forward(batch.float())

        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        negloglik_loss = torch.mean(negloglik_loss)

        negloglik_loss.backward()
        train_loss += negloglik_loss.item()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = np.sum(train_loss) / len(train_loader)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss


def val_maf(model, train, val_loader):
    model.eval()
    val_loss = []
    _, _ = model.forward(train.float())
    for batch in val_loader:
        u, log_det = model.forward(batch.float())
        print("u:", u.max(), u.min())
        print(log_det.max(), log_det.min())
        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        val_loss.extend(negloglik_loss.tolist())

    N = len(val_loader.dataset)
    print(
        "Validation loss: {:.4f} +/- {:.4f}".format(
            np.sum(val_loss) / N, 2 * np.std(val_loss) / np.sqrt(N)
        )
    )
    return np.sum(val_loss) / N


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
    i = 0
    N = len(test_loss)
    print(
        "Test loss: {:.4f} +/- {:.4f}".format(
            np.mean(test_loss), 2 + np.std(test_loss) / np.sqrt(N)
        )
    )


def sample_digits_maf(model, epoch, random_order=False, seed=None, test=False):
    model.eval()
    n_samples = 80
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    if random_order is True:
        np.random.seed(seed)
        order = np.random.permutation(784)
    else:
        order = np.arange(784)

    u = torch.zeros(800, 784).normal_(0, 1)
    mvn = MultivariateNormal(torch.zeros(28 * 28), torch.eye(28 * 28))
    dens = mvn.log_prob(u)
    samples, log_det = model.backward(u)
    log_det = dens - log_det
    log_det = log_det[np.logical_not(np.isnan(log_det.detach().numpy()))]

    idx = np.argsort(log_det.detach().numpy())
    samples = samples[idx].flip(dims=(0,))
    samples = samples[80 : 80 + n_samples]

    # idx = np.argsort(log_det.detach().numpy())[::-1][:n_samples]
    # idx.sort()
    # idx = idx - np.zeros_like(idx)
    # print(idx)
    # samples = samples[idx, :]
    samples = (torch.sigmoid(samples) - 1e-6) / (1 - 2e-6)
    samples = samples.detach().cpu().view(n_samples, 28, 28)

    fig, axes = plt.subplots(ncols=10, nrows=8)
    ax = axes.ravel()
    for i in range(80):
        ax[i].imshow(
            np.transpose(samples[i], (0, 1)), cmap="gray", interpolation="none"
        )
        ax[i].axis("off")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_frame_on(False)

    if not os.path.exists("gif_results"):
        os.makedirs("gif_results")
    if test is False:
        save_path = "gif_results/samples_gaussian_" + str(epoch) + ".pdf"
    else:
        save_path = "results/samples_gaussian_" + str(epoch) + ".pdf"

    fig.subplots_adjust(wspace=-0.35, hspace=0.065)
    plt.gca().set_axis_off()
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()


def plot_losses(epochs, train_losses, val_losses, title=None):
    sns.set(style="white")
    fig, axes = plt.subplots(
        ncols=1, nrows=1, figsize=[10, 5], sharey=True, sharex=True, dpi=400
    )

    train = pd.Series(train_losses).astype(float)
    val = pd.Series(val_losses).astype(float)
    train.index += 1
    val.index += 1

    axes = sns.lineplot(data=train, color="gray", label="Training loss")
    axes = sns.lineplot(data=val, color="orange", label="Validation loss")

    axes.set_ylabel("Negative log-likelihood")
    axes.legend(
        frameon=False,
        prop={"size": 14},
        fancybox=False,
        handletextpad=0.5,
        handlelength=1,
    )
    axes.set_ylim(1000, 1600)
    axes.set_xlim(0, 100)
    axes.set_title(title) if title is not None else axes.set_title(None)
    if not os.path.exists("results"):
        os.makedirs("results")
    save_path = "results/train_plots" + str(epochs[-1]) + ".pdf"
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()


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
