import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class L1L2Criterion(nn.Module):
    def __init__(self, alpha, beta):
        super(L1L2Criterion, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y):
        assert len(x.shape) == 2 and x.shape == y.shape
        sub = x - y
        sub.abs_()
        l1 = torch.sum(sub, 1)
        l = l1.mean().unsqueeze_(0).mul_(self.alpha)
        sub.pow_(2)
        l2 = torch.sum(sub, 1).sqrt_()
        l.add_(l2.mean().mul_(self.beta))
        return l
