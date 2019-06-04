import shutil

import torch
from torch import nn as nn, optim as optim

from src import settings, hyperparameters as hyperparams
from src.perceptual_loss import VGGDistance


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

        return l


def saveHyperParams(dest_path):
    shutil.copy(settings.p / 'src' / 'hyperparameters.py', dest_path)


def addNoise(x, mean, std):
    return x.add(torch.empty_like(x).normal_(mean=mean, std=std))


def findOptimalLatentVector(glo, image):
    res = torch.rand(hyperparams.latentDim, device=settings.device, requires_grad=True)
    criterion = VGGDistance(hyperparams.gloLossAlpha, hyperparams.gloLossBeta, hyperparams.gloLossPowAlpha,
                            hyperparams.gloLossPowBeta).to(settings.device)
    optimizer = optim.Adam([res], lr=hyperparams.gloEvalAdamLr, betas=hyperparams.gloEvalAdamBetas)

    for epoch in range(1, hyperparams.gloEvalEpochsNum):
        optimizer.zero_grad()
        loss = criterion(image.unsqueeze(0), glo(res.view(1, hyperparams.latentDim, 1, 1)))
        loss.backward()
        optimizer.step()

    return torch.empty(hyperparams.latentDim).to(settings.device).copy_(res)
