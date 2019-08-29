import math
import pickle
import shutil

import torch
import torch.nn as nn
from torch import optim as optim
from tqdm import tqdm

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


class L2Criterion(nn.Module):
    def __init__(self):
        super(L2Criterion, self).__init__()

    @staticmethod
    def l2(x, y):
        assert len(x.shape) == 2 and x.shape == y.shape
        sub = x - y
        sub.pow_(2)
        l = torch.sum(sub, 1)  # .sqrt_()
        return l.mean()

    def forward(self, x, y):
        l = self.l2(x, y)
        return l


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
    res = torch.empty(hyperparams.latentDim, device=settings.device, requires_grad=True)
    nn.init.normal_(res)
    criterion = VGGDistance(hyperparams.gloLossAlpha, hyperparams.gloLossBeta, hyperparams.gloLossPowAlpha,
                            hyperparams.gloLossPowBeta).to(settings.device)
    optimizer = optim.Adam([res], lr=hyperparams.gloEvalAdamLr, betas=hyperparams.gloEvalAdamBetas)

    for epoch in tqdm(range(1, hyperparams.gloEvalEpochsNum)):
        optimizer.zero_grad()
        loss = criterion(image.unsqueeze(0), glo(res.view(1, hyperparams.latentDim, 1, 1)))
        if epoch >= hyperparams.gloEvalEpochsNum - 30:
            print(loss.item())
        loss.backward()
        optimizer.step()

    return torch.empty(hyperparams.latentDim, device=settings.device).copy_(res)


def projectRowsToLpBall(mat, p=2):
    assert len(mat.shape) == 2
    norms = mat.norm(p=p, dim=1, keepdim=True)
    norms.clamp_(1, math.inf)
    mat.div_(norms)


def findNearest(A, B):
    assert len(A.shape) == 2 and len(B.shape) == 2 and A.shape[1] == B.shape[1]
    l = B.shape[0]
    ret = torch.empty(l, dtype=torch.int64, device=settings.device)
    for i in range(l):
        distancesSquared = A - B[i]
        distancesSquared = distancesSquared.pow_(2).sum(1)
        ret[i] = distancesSquared.argmin()

    return ret


def loadPickle(path, mode='rb'):
    with open(path, mode) as f:
        pcklr = pickle.Unpickler(f)
        ret = pcklr.load()
    return ret


def storePickle(path, obj, mode='wb'):
    with open(path, mode) as f:
        pcklr = pickle.Pickler(f)
        pcklr.dump(obj)
