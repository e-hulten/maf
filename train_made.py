import torch
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import os

from made import MADE
from model import MAFLayer
from utils_maf import train_one_epoch_made, val_made, test_made

# --------- parameters ----------
dataset = "power"
n_in = 784
hidden_dims = [100, 100]
lr = 1e-3
epochs = 0
seed = 19
patience = 30
# -------------------------------

if dataset == "mnist":
    from data.mnist import train, train_loader, val_loader, test_loader, n_in
elif dataset == "power":
    from data.power import train, train_loader, val_loader, test_loader, n_in
elif dataset == "gas":
    from gas import train, train_loader, val_loader, test_loader, n_in
else:
    raise ValueError(
        "Unknown dataset...\n\nPlease choose between 'mnist','power','gas', and 'hepmass'."
    )

model = MADE(n_in, hidden_dims, random_order=False, seed=seed, gaussian=True)
model = torch.load("models/made_100_100.pt")
test_made(model, test_loader)


print(
    "Number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()])
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

string = ""
for i in range(len(hidden_dims)):
    string += "_" + str(hidden_dims[i])

i = 0
max_loss = np.inf
# for plotting
epochs_list = []
train_losses = []
val_losses = []

for epoch in range(epochs):
    train_one_epoch_made(model, epoch, optimizer, train_loader)
    val_loss = val_made(model, val_loader)
    if val_loss < max_loss:
        max_loss = val_loss
        i = 0
        torch.save(model, "./models/made" + string + ".pt")
    else:
        i += 1
    if i >= patience:
        print("Patience counter: {}/{}\n Terminate training!".format(i, patience))
        break
    print("Patience counter: {}/{}".format(i, patience))

