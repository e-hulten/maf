import torch
from data.mnist import MNIST
from data.power import POWER
from data.hepmass import HEPMASS


def data_loader(dataset, batch_size=128):
    if dataset == "mnist":
        data = MNIST(logit=True, dequantize=True)
    elif dataset == "power":
        data = POWER()
    elif dataset == "hepmass":
        data = HEPMASS()
    else:
        raise ValueError(
            'Unknown dataset. Please choose between "mnist", "power", and "hepmass".'
        )

    train = torch.from_numpy(data.train.x)
    val = torch.from_numpy(data.val.x)
    test = torch.from_numpy(data.test.x)
    n_in = data.n_dims

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,)

    return train, train_loader, val_loader, test_loader, n_in

