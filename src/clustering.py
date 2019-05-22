from src import settings
import torch

from src.frogsDataset import FrogsDataset as Dataset


def dataMatrix(dataset):
    images = []
    for sample in iter(dataset):
        images.append(sample['image'])
    # TODO stack.... maybe flatten?
    return images


def kMeans(points):
    ...


def PCA(dataset, dim):
    X = dataMatrix(dataset)
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


if __name__ == '__main__':
    dataset = Dataset(settings.frogs, settings.frogs1000)
    A = dataMatrix(dataset)
