import shutil

import torch
import torch.nn as nn
from torch import optim as optim
from tqdm import tqdm

from src import settings, hyperparameters as hyperparams
from src.frogsDataset import FrogsDataset as Dataset
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

    return torch.empty(hyperparams.latentDim).to(settings.device).copy_(res)


def indMappings(datasetSubset, dataset):
    fileind_to_mainind = {dataset[i]['fileind'].item(): i for i in range(len(dataset))}
    return {i: fileind_to_mainind[datasetSubset[i]['fileind'].item()] for i in range(len(datasetSubset))}


def mergeEmbeddings(dataset1, dataset2, dataset, embed1Path, embed2Path, resPath):
    dsize1 = len(dataset1)
    dsize2 = len(dataset2)
    dsize = len(dataset)
    assert dsize == dsize1 + dsize2

    embed1 = nn.Embedding(dsize1, hyperparams.latentDim).to(settings.device)
    embed2 = nn.Embedding(dsize2, hyperparams.latentDim).to(settings.device)
    res = nn.Embedding(dsize, hyperparams.latentDim).to(settings.device)
    embed1.load_state_dict(torch.load(embed1Path))
    embed2.load_state_dict(torch.load(embed2Path))
    embed1.eval()
    embed2.eval()
    res.eval()

    ind1_to_ind = indMappings(dataset1, dataset)
    ind2_to_ind = indMappings(dataset2, dataset)
    matrix = torch.empty((dsize, hyperparams.latentDim))

    with torch.no_grad():
        for d, indMapping, embed in zip((dataset1, dataset2), (ind1_to_ind, ind2_to_ind), (embed1, embed2)):
            for i in range(len(d)):
                matrix[indMapping[i]] = embed(torch.tensor([i]).to(settings.device)).view(1, hyperparams.latentDim)

        res.weight.data.copy_(matrix)

    torch.save(res.state_dict(), resPath)


if __name__ == '__main__':
    dataset1 = Dataset(settings.frogs, settings.frogsSubset1)
    dataset2 = Dataset(settings.frogs, settings.frogsSubset2)
    dataset = Dataset(settings.frogs, settings.frogsSubset)
    embed1Path = settings.localModels / 'glo1/latent.pt'
    embed2Path = settings.localModels / 'glo2/latent.pt'
    resPath = settings.localModels / 'enc2/latent.pt'
    mergeEmbeddings(dataset1, dataset2, dataset, embed1Path, embed2Path, resPath)
