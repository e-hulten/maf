import torch
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from data.power import test_loader, train
from model import MAF
from utils_maf import (
    val_maf,
    test_maf,
    sample_digits_maf,
)

model = torch.load("models/maf_power_100_100.pt")
test_maf(model, train, test_loader)
# val_maf(model, 0)
# print("sample digits")
# sample_digits_maf(model, "test")

plot = False
if plot:
    model.eval()
    batch = next(iter(test_loader))
    u = model(batch)[0].detach().numpy()
    fig, axes = plt.subplots(
        ncols=6, nrows=4, sharex=True, sharey=True, figsize=(16, 10)
    )

    for ax in axes.reshape(-1):
        dim1 = np.random.randint(28 * 28)
        dim2 = np.random.randint(28 * 28)
        ax.scatter(u[:, dim1], u[:, dim2], color="black", s=0.5)
        ax.set_ylabel("dim: " + str(dim2), size=14)
        ax.set_xlabel("dim: " + str(dim1), size=14)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect(1)

    plt.savefig("scatter.png", bbox_inches="tight", dpi=300)
    plt.savefig("scatter.pdf", bbox_inches="tight", dpi=300)

    fig, axes = plt.subplots(
        ncols=6, nrows=4, sharex=True, sharey=True, figsize=(16, 10)
    )

    for ax in axes.reshape(-1):
        dim1 = np.random.randint(28 * 28)
        sns.distplot(u[:, dim1], ax=ax, color="black")
        ax.set_xlabel("dim: " + str(dim1), size=14)
        ax.set_xlim(-5, 5)

    plt.savefig("marginal.png", bbox_inches="tight", dpi=300)
    plt.savefig("marginal.pdf", bbox_inches="tight", dpi=300)

    sample_digits_maf(model, "test")

