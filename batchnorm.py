import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))
        self.batch_mean = None
        self.batch_var = None

    def forward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps  # torch.mean((x - m) ** 2, axis=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean.clone()
            v = self.batch_var.clone()

        x_hat = (x - m) / torch.sqrt(v)
        x_hat = x_hat * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma - 0.5 * torch.log(v))
        return x_hat, log_det

    def backward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean
            v = self.batch_var

        x_hat = (x - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x_hat, log_det

    def set_batch_stats_func(self, x):
        print("setting batch stats for validation")
        self.batch_mean = x.mean(dim=0)
        self.batch_var = x.var(dim=0) + self.eps


class BatchNorm_running(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.momentum = 0.01
        self.gamma = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.running_mean = torch.zeros(1, dim)
        self.running_var = torch.ones(1, dim)

    def forward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps  # torch.mean((x - m) ** 2, axis=0) + self.eps
            self.running_mean *= 1 - self.momentum
            self.running_mean += self.momentum * m
            self.running_var *= 1 - self.momentum
            self.running_var += self.momentum * v
        else:
            m = self.running_mean
            v = self.running_var

        x_hat = (x - m) / torch.sqrt(v)
        x_hat = x_hat * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma) - 0.5 * torch.sum(torch.log(v))
        return x_hat, log_det

    def backward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps
            self.running_mean *= 1 - self.momentum
            self.running_mean += self.momentum * m
            self.running_var *= 1 - self.momentum
            self.running_var += self.momentum * v
        else:
            m = self.running_mean
            v = self.running_var

        x_hat = (x - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x_hat, log_det
