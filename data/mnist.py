import torch
import torch.utils.data
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

""" 
This is a version of: https://github.com/gpapamak/maf/blob/master/datasets/mnist.py, 
adapted to work with Python 3.x and PyTorch. 
"""

batch_size = 100


class MNIST:

    alpha = 1e-6

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data, logit, dequantize, rng):

            x = (
                self._dequantize(data[0], rng) if dequantize else data[0]
            )  # dequantize pixels
            self.x = self._logit_transform(x) if logit else x  # logit
            self.N = self.x.shape[0]  # number of datapoints

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return x + rng.rand(*x.shape) / 256.0

        @staticmethod
        def _logit_transform(x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            x = MNIST.alpha + (1 - 2 * MNIST.alpha) * x
            return np.log(x / (1.0 - x))

    def __init__(self, logit=True, dequantize=True):
        root = "data/maf_data/"
        # load dataset
        f = gzip.open(root + "mnist/mnist.pkl.gz", "rb")
        train, val, test = pickle.load(f, encoding="latin1")
        f.close()

        rng = np.random.RandomState(42)
        self.train = self.Data(train, logit, dequantize, rng)
        self.val = self.Data(val, logit, dequantize, rng)
        self.test = self.Data(test, logit, dequantize, rng)

        self.n_dims = self.train.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims))] * 2

    def show_pixel_histograms(self, split, pixel=None):
        """
        Shows the histogram of pixel values, or of a specific pixel if given.
        """

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError("Invalid data split")

        if pixel is None:
            data = data_split.x.flatten()

        else:
            row, col = pixel
            idx = row * self.image_size[0] + col
            data = data_split.x[:, idx]

        n_bins = int(np.sqrt(data_split.N))
        fig, ax = plt.subplots(1, 1)
        ax.hist(data, n_bins, density=True, color="lightblue")
        ax.set_yticklabels("")
        ax.set_yticks([])
        plt.show()

