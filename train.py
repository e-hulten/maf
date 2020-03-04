import torch
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import os
from model import MAF, MAFLayer, BatchNorm
from utils_maf import (
    train_one_epoch_maf,
    val_maf,
    test_maf,
    sample_digits_maf,
    plot_losses,
)

# --------- parameters ----------
dataset = "hepmass"
n_mades = 5  # number of layers in the MAF
hidden_dims = [512]  # the size of each MADE
lr = 1e-4
epochs = 100000  # use early stopping
random_order = False
patience = 30
# -------------------------------

if dataset == "mnist":
    from data.mnist import train, train_loader, val_loader, test_loader, n_in
elif dataset == "power":
    from data.power import train, train_loader, val_loader, test_loader, n_in
elif dataset == "hepmass":
    from data.hepmass import train, train_loader, val_loader, test_loader, n_in
else:
    raise ValueError(
        "Unknown dataset...\n\nPlease choose between 'mnist','power', and 'hepmass'."
    )


model = MAF(n_in, n_mades, hidden_dims)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
print(
    "Number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()])
)

string = "_" + str(dataset)
for i in range(len(hidden_dims)):
    string += "_" + str(hidden_dims[i])

i = 0
max_loss = np.inf
# for plotting
epochs_list = []
train_losses = []
val_losses = []

if not os.path.exists("models"):
    os.makedirs("models")

for epoch in range(1, epochs + 1):
    epochs_list.append(epoch)
    train_loss = train_one_epoch_maf(model, epoch, optimizer, train_loader)
    train_losses.append(train_loss)
    val_loss = val_maf(model, train, val_loader)
    val_losses.append(val_loss)

    if val_loss < max_loss:
        max_loss = val_loss
        i = 0
        torch.save(model, "./models/maf" + string + ".pt")
    else:
        i += 1
    if i >= patience:
        print("Patience counter: {}/{}\n Terminate training!".format(i, patience))
        break
    print("Patience counter: {}/{}".format(i, patience))

