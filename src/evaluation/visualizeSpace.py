import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src import hyperparameters as hyperparams
from src import settings
from src.clustering import pca
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.encoder import Encoder


def vis(lowDimMat, color):
    x = lowDimMat[:, 0]
    y = lowDimMat[:, 1]

    plt.scatter(x, y, c=color, alpha=0.75, marker='o')


def lowDim(mat):
    mat = torch.from_numpy(mat)
    encMat = pca.PCA(mat, hyperparams.clusteringPCADim)
    mat = pca.encodeMat(mat, encMat)
    return mat.cpu().numpy()


def genLatent(enc, dloader):
    latentVecs = []
    for batch in dloader:
        with torch.no_grad():
            images = batch['image'].to(settings.device).type(torch.float32)
            latentVecs.append(enc(images))
    return torch.cat(latentVecs, 0)


def main():
    subset1 = Dataset(settings.frogs, settings.frogsSubset1)
    subset2 = Dataset(settings.frogs, settings.frogsSubset2)
    main = Dataset(settings.frogs, settings.frogsMain)
    dloader_s1 = DataLoader(subset1, batch_size=settings.clusteringBatchSize, shuffle=False)
    dloader_s2 = DataLoader(subset2, batch_size=settings.clusteringBatchSize, shuffle=False)
    dloader_m = DataLoader(main, batch_size=settings.clusteringBatchSize, shuffle=False)
    enc = Encoder().to(settings.device)

    subset1Latent = genLatent(enc, dloader_s1)
    subset2Latent = genLatent(enc, dloader_s2)
    mainLatent = genLatent(enc, dloader_m)

    subset1LatentLow = lowDim(subset1Latent)
    subset2LatentLow = lowDim(subset2Latent)
    mainLatentLow = lowDim(mainLatent)

    vis(subset1LatentLow, 'red')
    vis(subset2LatentLow, 'blue')
    vis(mainLatentLow, 'green')

    assert not settings.spaceVisPath.is_file()
    # plt.savefig(settings.spaceVisPath, dpi=600)
    plt.show()

if __name__ == '__main__':
    main()
