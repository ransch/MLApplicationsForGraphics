import pickle

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from src import hyperparameters as hyperparams
from src import settings
from src.clustering import pca


def load(pcaPath, clusteringPath, representativesPath):
    with open(pcaPath, 'rb') as f:
        pcklr = pickle.Unpickler(f)
        lowDimMat = pcklr.load()

    with open(clusteringPath, 'rb') as f:
        pcklr = pickle.Unpickler(f)
        buckets, _ = pcklr.load()

    with open(representativesPath, 'rb') as f:
        pcklr = pickle.Unpickler(f)
        representatives = pcklr.load()

    return lowDimMat, buckets, representatives


def vis(lowDimMat, buckets, representatives):
    labels = []

    for i in range(lowDimMat.shape[0]):
        l = -1
        for k, v in buckets.items():
            if i in v:
                l = k
                break
        assert l >= 0
        labels.append(l)

    x = lowDimMat[:, 0]
    y = lowDimMat[:, 1]
    reprInds = []
    for lst in representatives.values():
        reprInds.extend(lst)

    reprMat = lowDimMat[reprInds, :]
    reprX = reprMat[:, 0]
    reprY = reprMat[:, 1]

    plt.scatter(x, y, c=labels, alpha=0.75, marker='o')
    plt.scatter(reprX, reprY, c="red", alpha=0.75, marker='X')
    # plt.show()
    print(settings.localModels / 'repr-dim50-128-pca.jpg')
    plt.savefig(settings.localModels / 'repr-dim50-128-pca.jpg', dpi=600)


def lowDim(method, mat):
    if method == 'pca':
        mat = torch.from_numpy(mat)
        encMat = pca.PCA(mat, hyperparams.clusteringPCADim)
        mat = pca.encodeMat(mat, encMat)
        return mat.cpu().numpy()
    assert method == 'pca'
    return TSNE(n_components=2).fit_transform(mat)


def main():
    lowDimMat, buckets, representatives = load(settings.pcaPath, settings.clusteringPath, settings.representativesPath)

    if lowDimMat.shape[1] > 2:
        lowDimMat = lowDim('pca', lowDimMat)

    vis(lowDimMat, buckets, representatives)


if __name__ == '__main__':
    main()
