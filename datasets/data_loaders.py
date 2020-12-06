import torch
from .mnist import MNISTDataset
from .power import PowerDataset
from .hepmass import HEPMassDataset


def get_data(dataset: str):
    dataset = dataset.lower()
    if dataset == "mnist":
        return MNISTDataset(logit=True, dequantize=True)
    if dataset == "power":
        return PowerDataset()
    if dataset == "hepmass":
        return HEPMassDataset()

    raise ValueError(
        f"Unknown dataset '{dataset}'. Please choose either 'mnist', 'power', or 'hepmass'."
    )


def get_data_loaders(data, batch_size: int = 128):
    train = torch.from_numpy(data.train.x)
    val = torch.from_numpy(data.val.x)
    test = torch.from_numpy(data.test.x)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,)

    return train_loader, val_loader, test_loader
