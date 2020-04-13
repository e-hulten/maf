import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import math


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
    axes.set_ylim(1250, 1600)
    axes.set_xlim(0, 50)
    axes.set_title(title) if title is not None else axes.set_title(None)
    if not os.path.exists("plots"):
        os.makedirs("plots")
    save_path = "plots/train_plots" + str(epochs[-1]) + ".pdf"
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()
