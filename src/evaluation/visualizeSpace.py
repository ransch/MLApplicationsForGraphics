import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src import settings
from src.clustering import pca
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.encoder import Encoder


def vis(lowDimMat, color):
    x = lowDimMat[:, 0]
    y = lowDimMat[:, 1]

    plt.scatter(x, y, c=color, alpha=0.65, marker='o')


def lowDim(mat):
    encMat = pca.PCA(mat, 2)
    return pca.encodeMat(mat, encMat).cpu().numpy()


def genLatent(enc, dloader):
    latentVecs = []
    for batch in dloader:
        with torch.no_grad():
            images = batch['image'].to(device=settings.device, dtype=torch.float32)
            latentVecs.append(enc(images))
    return torch.cat(latentVecs, 0)


def main():
    visPath = settings.matureModels / 'arch/arch4/latVis.jpg'
    encPath = settings.matureModels / 'arch/arch4/enc.pt'

    subset1 = Dataset(settings.frogs, settings.frogsSubset1)
    subset2 = Dataset(settings.frogs, settings.frogsSubset2)
    main = Dataset(settings.frogs, settings.frogsMain)
    dloader_s1 = DataLoader(subset1, batch_size=settings.bigBatchSize, shuffle=False)
    dloader_s2 = DataLoader(subset2, batch_size=settings.bigBatchSize, shuffle=False)
    dloader_m = DataLoader(main, batch_size=settings.bigBatchSize, shuffle=False)
    enc = Encoder().to(settings.device)
    enc.load_state_dict(torch.load(encPath))
    enc.eval()

    subset1Latent = genLatent(enc, dloader_s1)
    subset2Latent = genLatent(enc, dloader_s2)
    mainLatent = genLatent(enc, dloader_m)

    subset1LatentLow = lowDim(subset1Latent)
    subset2LatentLow = lowDim(subset2Latent)
    mainLatentLow = lowDim(mainLatent)

    vis(mainLatentLow, 'black')
    vis(subset2LatentLow, 'blue')
    vis(subset1LatentLow, 'red')

    assert not visPath.is_file()
    plt.savefig(visPath, dpi=600)
    # plt.show()


if __name__ == '__main__':
    main()
