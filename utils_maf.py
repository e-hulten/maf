import torch
from torch.distributions import MultivariateNormal

import matplotlib.pyplot as plt
import os
import numpy as np
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

    u = torch.zeros(n_samples, 784).normal_(0, 1)
    mvn = MultivariateNormal(torch.zeros(28 * 28), torch.eye(28 * 28))
    log_prob = mvn.log_prob(u)
    samples, log_det = model.backward(u)

    # log_det = log_prob - log_det
    # log_det = log_det[np.logical_not(np.isnan(log_det.detach().numpy()))]
    # idx = np.argsort(log_det.detach().numpy())
    # samples = samples[idx].flip(dims=(0,))
    # samples = samples[80 : 80 + n_samples]

    samples = (torch.sigmoid(samples) - 1e-6) / (1 - 2e-6)
    samples = samples.detach().cpu().view(n_samples, 28, 28)

    fig, axes = plt.subplots(ncols=10, nrows=8)
    ax = axes.ravel()
    for i in range(n_samples):
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
        save_path = "gif_results/samples_gaussian_" + str(epoch) + ".png"
    else:
        save_path = "figs/samples_gaussian_" + str(epoch) + ".png"

    fig.subplots_adjust(wspace=-0.35, hspace=0.065)
    plt.gca().set_axis_off()
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()

