import pickle

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset


def dataMatrix(dloader):
    features = []
    resnet = models.resnet50(pretrained=True).to(settings.device)
    resnet.fc = nn.Identity()
    resnet.eval()

    for batch in dloader:
        with torch.no_grad():
            images = batch['image'].to(settings.device).type(torch.float32)
            features.append(resnet(images))
    return torch.cat(features)


def PCA(X, dim):
    samples = X.shape[0]
    origDim = X.shape[1]
    assert samples > origDim

    with torch.no_grad():
        A = X.t() @ X
        eig, V = torch.symeig(A, eigenvectors=True)
        _, inds = torch.topk(eig, dim, largest=True, sorted=True)
        topVecs = []
        for ind in inds:
            topVecs.append(V[:, ind])

    res = torch.stack(topVecs, dim=0)
    assert res.shape == (dim, origDim)
    return res


def encodeMat(mat, encMat):
    return mat @ encMat.t()


def main():
    settings.sysAsserts()
    settings.pcaAsserts()
    dataset = Dataset(settings.frogs, settings.frogs6000)
    dloader = DataLoader(dataset, batch_size=settings.clusteringBatchSize, shuffle=False)

    X = dataMatrix(dloader)
    encMat = PCA(X, hyperparams.clusteringPCADim)
    lowDimMat = encodeMat(X, encMat)
    lowDimMat = lowDimMat.cpu().numpy()

    with open(settings.pcaPath, 'wb') as f:
        pcklr = pickle.Pickler(f)
        pcklr.dump(lowDimMat)


if __name__ == '__main__':
    main()
