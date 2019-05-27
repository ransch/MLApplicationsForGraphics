import pickle

import torch
import torch.nn as nn
import torchvision.models as torchModels

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset


def dataMatrix(dataset):
    features = []
    resnet = torchModels.resnet50(pretrained=True).to(settings.device)
    resnet.fc = nn.Identity()
    resnet.eval()

    for i in range(len(dataset)):
        image = dataset[i]['image'].to(settings.device).type(torch.float32)
        features.append(resnet(image.unsqueeze_(0)))
    return torch.cat(features)


def PCA(X, reducedDim):
    samples_len = X.shape[0]
    origDim = X.shape[1]
    assert samples_len > origDim

    with torch.no_grad():
        A = X.t() @ X
        eig, V = torch.symeig(A, eigenvectors=True)
        _, inds = torch.topk(eig, k=reducedDim, largest=True, sorted=True)  # biggest to smallest
        topVecs = []
        for ind in inds:
            topVecs.append(V[ind])

    resMat = torch.stack(topVecs, dim=0)  # W matrix (from book)
    assert resMat.shape == (reducedDim, origDim)  # (n*d)
    return resMat


def encodeMat(mat, encMat):
    return mat @ encMat.t()  # checked!


def main():
    settings.sysAsserts()
    settings.pcaAsserts()
    dataset = Dataset(settings.frogs, settings.frogs6000)

    X = dataMatrix(dataset)
    encMat = PCA(X, hyperparams.clusteringPCADim)
    lowDimMat = encodeMat(X, encMat)
    lowDimMat = lowDimMat.cpu().numpy()

    with open(settings.pcaPath, 'wb') as f:
        pcklr = pickle.Pickler(f)
        pcklr.dump(lowDimMat)


if __name__ == '__main__':
    main()
