import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os

# download binarised mnist: https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz
# path to binarized_mnist.npz:
path = "../made/data/binarized_mnist.npz"
batch_size = 128

mnist = np.load(path)

train, test = mnist["train_data"], mnist["test_data"]
train = torch.from_numpy(train)
test = torch.from_numpy(test)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True,)

