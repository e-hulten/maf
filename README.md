# Masked Autoregressive Flow with PyTorch

This is a PyTorch implementation of the masked autoregressive flow (MAF) by Papamakarios et al. [1]. The Gaussian MADE that makes up each layer in the MAF is found in ``MADE.py``, while the MAF itself is found in ``model.py``.

The files in the ``data`` folder are adapted from the original repository by G. Papamakarios [2]. The datasets can be downloaded following the link in his repository. 

**Note:** The validation loss diverges for the first few epochs, before it stabilises and gets on the right track. Might have to check my weight initialisation, which is different from the one used in the paper. 

[1] https://arxiv.org/abs/1705.07057
[2] https://github.com/gpapamak/maf/blob/master/datasets/
