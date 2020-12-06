import torch
import numpy as np
from maf import MAF
from made import MADE
from datasets.data_loaders import get_data, get_data_loaders
from utils.train import train_one_epoch_maf, train_one_epoch_made
from utils.validation import val_maf, val_made
from utils.test import test_maf, test_made
from utils.plot import sample_digits_maf, plot_losses


# --------- SET PARAMETERS ----------
model_name = "maf"  # 'MAF' or 'MADE'
dataset_name = "mnist"
batch_size = 128
n_mades = 5
hidden_dims = [512]
lr = 1e-4
random_order = False
patience = 30  # For early stopping
seed = 290713
plot = True
max_epochs = 1000
# -----------------------------------

# Get dataset.
data = get_data(dataset_name)
train = torch.from_numpy(data.train.x)
# Get data loaders.
train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
# Get model.
n_in = data.n_dims
if model_name.lower() == "maf":
    model = MAF(n_in, n_mades, hidden_dims)
elif model_name.lower() == "made":
    model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True)
# Get optimiser.
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)


# Format name of model save file.
save_name = f"{model_name}_{dataset_name}_{'_'.join(str(d) for d in hidden_dims)}.pt"
# Initialise list for plotting.
epochs_list = []
train_losses = []
val_losses = []
# Initialiise early stopping.
i = 0
max_loss = np.inf
# Training loop.
for epoch in range(1, max_epochs):
    if model_name == "maf":
        train_loss = train_one_epoch_maf(model, epoch, optimiser, train_loader)
        val_loss = val_maf(model, train, val_loader)
    elif model_name == "made":
        train_loss = train_one_epoch_made(model, epoch, optimiser, train_loader)
        val_loss = val_made(model, val_loader)
    if plot:
        sample_digits_maf(model, epoch, random_order=random_order, seed=5)

    epochs_list.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Early stopping. Save model on each epoch with improvement.
    if val_loss < max_loss:
        i = 0
        max_loss = val_loss
        torch.save(
            model, "model_saves/" + save_name
        )  # Will print a UserWarning 1st epoch.
    else:
        i += 1

    if i < patience:
        print("Patience counter: {}/{}".format(i, patience))
    else:
        print("Patience counter: {}/{}\n Terminate training!".format(i, patience))
        break

plot_losses(epochs_list, train_losses, val_losses, title=None)
