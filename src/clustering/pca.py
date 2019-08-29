import torch
import torch.nn as nn
import torchvision.models as torchModels
from torch.utils.data import DataLoader

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.utils import loadPickle, storePickle


def dataMatrix(dloader):
    features = []
    resnet = torchModels.resnet50(pretrained=True).to(settings.device)
    resnet.fc = nn.Identity()
    resnet.eval()
    last_ind = 0

    for batch in dloader:
        with torch.no_grad():
            indices = batch['ind'].to(device=settings.device, dtype=torch.float32).squeeze_(1)
            for ind in indices:
                assert ind == last_ind
                last_ind += 1
            images = batch['image'].to(device=settings.device, dtype=torch.float32)
            features.append(resnet(images))
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


def reduceDim(dloader, pcaPath):
    X = dataMatrix(dloader)
    encMat = loadPickle(pcaPath)
    return encodeMat(X, encMat).cpu().numpy()


def main():
    settings.sysAsserts()
    settings.pcaAsserts()
    dataset = Dataset(settings.frogs, settings.frogs6000)
    dloader = DataLoader(dataset, batch_size=settings.bigBatchSize, shuffle=False)

    X = dataMatrix(dloader)
    encMat = PCA(X, hyperparams.clusteringPCADim)
    storePickle(settings.pcaPath, encMat)


if __name__ == '__main__':
    main()
