import torch
import numpy as np
import os
import sys
import argparse

from data.loader import data_loader
from maf import MAF
from made import MADE
from utils_maf import (
    train_one_epoch_maf,
    val_maf,
    test_maf,
    sample_digits_maf,
)
from utils_made import train_one_epoch_made, val_made, test_made
from utils import plot_losses

parser = argparse.ArgumentParser(description="Set hyperparameters for training MAF.")
parser.add_argument(
    "--model", default="maf", type=str, help='Choose model: "maf" or "made"'
)
parser.add_argument("--data", default="mnist", type=str, help="Choose dataset.")
parser.add_argument("--batch_size", default=128, type=int, help="Choose batch size.")
parser.add_argument(
    "--n_mades", default=5, type=int, help="Number of layers (MADEs) in the MAF."
)
parser.add_argument(
    "--hidden_dims",
    default=[512],
    type=int,
    nargs="+",
    help="Size of each MADE. Integers represent layer sizes, and should be separated by spaces.",
)
parser.add_argument(
    "--lr",
    default=1e-4,
    type=float,
    help="Choose learning rate for the Adam optimiser.",
)
parser.add_argument(
    "--random_order",
    default=False,
    type=bool,
    help="Choose order of input. Default: False, i.e., natural ordering.",
)
parser.add_argument(
    "--patience", default=30, type=int, help="Patience for early stopping.",
)
parser.add_argument(
    "--plot",
    default=True,
    type=bool,
    help='Plot during training or not. Only relevant for the "mnist" dataset.',
)
args = parser.parse_args()

# --------- SET PARAMETERS ----------
dataset = args.data
batch_size = args.batch_size
n_mades = args.n_mades
hidden_dims = args.hidden_dims
lr = args.lr
random_order = args.random_order
patience = args.patience
seed = 290713
if args.plot:
    print(
        "WARNING: Plotting during training significantly slows down the training process.\nIf time is a concern, please consider setting --plot False."
    )
# -----------------------------------

# load data loaders
train, train_loader, val_loader, test_loader, n_in = data_loader(dataset, batch_size)

if args.model == "maf":
    model = MAF(n_in, n_mades, hidden_dims)
elif args.model == "made":
    model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

# creating directory for saving the model
if not os.path.exists("models"):
    os.makedirs("models")

# naming the model
string = args.model + "_" + dataset
for i in range(len(hidden_dims)):
    string += "_" + str(hidden_dims[i])
string += ".pt"

# for plotting
epochs_list = []
train_losses = []
val_losses = []

# for early stopping
i = 0
max_loss = np.inf

# training
for epoch in range(1, sys.maxsize):
    if args.model == "maf":
        train_loss = train_one_epoch_maf(model, epoch, optimizer, train_loader)
        val_loss = val_maf(model, train, val_loader)
    elif args.model == "made":
        train_loss = train_one_epoch_made(model, epoch, optimizer, train_loader)
        val_loss = val_made(model, val_loader)

    if args.plot:
        sample_digits_maf(model, epoch, random_order=False, seed=5)

    epochs_list.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < max_loss:
        i = 0
        max_loss = val_loss
        torch.save(model, "./models/" + string)  # will print a userwarning 1st epoch
    else:
        i += 1

    if i < patience:
        print("Patience counter: {}/{}".format(i, patience))
    else:
        print("Patience counter: {}/{}\n Terminate training!".format(i, patience))
        break
