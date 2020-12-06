# :christmas_tree: Masked Autoregressive Flow with PyTorch
This is a PyTorch implementation of the masked autoregressive flow (MAF) by Papamakarios et al. [1]. 

The Gaussian MADE that makes up each layer in the MAF is found in ``MADE.py``, while the MAF itself is found in ``maf.py``.

## Datasets
The files in the ``data`` folder are adapted from the original repository by G. Papamakarios [2]. G. Papamakarios et al. have kindly made the preprocessed datasets available to the public, and they can be downloaded through this link: https://zenodo.org/record/1161203#.Wmtf_XVl8eN. 

## Example
Remember to download the datasets first, and then run 
```
python3 train.py
```
This will train a five layer MAF on the MNIST dataset. The size of each MAF layer (i.e., each Gaussian MADE) is set to be one layer of 512 hidden units. Following the approach of the original paper [1], we use the natural ordering of the inputs, and reverse it after each MAF layer. The model is trained using the Adam optimiser with a learning rate of `1e-4` and early stopping with a patience of 30. Please have a look inside `train.py` to see the rest of the hyperparameters and their default values. 

By changing the number of hidden units in each MADE to 1024, the model will converge to a test log-likelihood similar to the one reported in [1]. Interestingly, fewer hidden units in each layers give better test results in terms of likelihood.

Do not hesitate to kill (ctrl + c) the training when the validation loss has increased for more than five consecutive epochs (don't worry, the best model (on val data) is already saved). In my experience, the validation loss will rarely start decreasing again after that. Alternatively, can manually change the patience by editing `train.py`. 

Descent sample quality for MNIST is achieved in ~10 epochs, but further training squeezes out more performance in terms of higher average log-likelihood at test time. 

The training runs smoothly when run locally on a Macbook Pro 2018 model with 16GB RAM. (Be prepared for a noisy fan and a burning hot laptop.)

## Visualisations
An animation for a MAF with default settings trained for 20 epochs (one frame per epoch) on the MNIST dataset. The validation loss after 20 epochs was `1299.3 +/- 1.6`, with error bands corresponding to two empirical standard deviations. After ~70 epochs, the validation loss was down to `1283.0 +/- 1.9`. 

![alt text](https://github.com/e-hulten/maf/blob/master/figs/maf_512_mnist.gif "Visualisation of 20 training epochs.")

Below are 80 random samples from the best saved model during training. Note that the samples are not sorted by likelihood, and that there are some garbage among them. 

![alt text](https://github.com/e-hulten/maf/blob/master/figs/samples_gaussian_test.png "Random samples from MAF.")

![alt text](https://github.com/e-hulten/maf/blob/master/figs/maf_mnist_512_marginal.png "Marginal distributions of 24 random pixels.")
This figure show the marginal distribution of 80 random pixels, based on 1000 random test samples. For a perfect data-to-noise mapping, we would expect each of the marginals to follow a standard Gaussian. This holds for some pixels, but it seems like the majority has a longer lower tail than desired. 

![alt text](https://github.com/e-hulten/maf/blob/master/figs/maf_mnist_512_scatter.png "Scatterplot.")
The same trend is reflected in these scatterplots, which under a perfect data-to-noise mapping would be bivariate standard Gaussians. However, the third quadrant is in many cases relatively overpopulated, reflecting the lower tails of the marginal distributions. 

## To do 
**To do:** The validation loss diverges for the first few epochs for some of the datasets (not MNIST), before it stabilises and gets on the right track. Check:
* The weight initialisation, which is different from the one used in the MAF paper. 
* The preprocessing of the datasets. 
* The batchnorm layer. 
Also, there are sometimes a (very) few test samples that are not well received by the model (even for MNIST). This causes the test likelihood to be artificially low, and its sample variance to be artificially high. Typically less than three samples  with likelihoods that are orders of magnitude different from the rest of the samples. Find out what causes this, and how to deal with it. 

[1] https://arxiv.org/abs/1705.07057

[2] https://github.com/gpapamak/maf/blob/master/datasets/
