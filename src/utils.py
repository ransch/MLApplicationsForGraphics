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


class L1L2PertCriterion(nn.Module):
    def __init__(self, alpha, beta, pertmean=0, pertstd=0, pertalpha=0, pertbeta=0, pertgamma=0):
        super(L1L2PertCriterion, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pertmean = pertmean
        self.pertstd = pertstd
        self.pertalpha = pertalpha
        self.pertbeta = pertbeta
        self.pertgamma = pertgamma

    @staticmethod
    def l1l2(x, y, alpha, beta):
        assert len(x.shape) == 2 and x.shape == y.shape
        sub = x - y
        sub.abs_()
        l1 = torch.sum(sub, 1)
        l = l1.mean().unsqueeze_(0).mul_(alpha)
        sub.pow_(2)
        l2 = torch.sum(sub, 1).sqrt_()
        l.add_(l2.mean().mul_(beta))
        return l

    def forward(self, x, y):
        l = self.l1l2(x, y, self.alpha, self.beta)

        if self.pertgamma > 0:
            y.add_(torch.empty_like(y).normal_(mean=self.pertmean, std=self.pertstd))
            l.add_(self.l1l2(x, y, self.pertalpha, self.pertbeta).mul_(self.pertgamma))

        return l
