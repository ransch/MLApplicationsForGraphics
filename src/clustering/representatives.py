import pickle

import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from src import hyperparameters as hyperparams
from src import settings
from src.clustering.pca import reduceDim
from src.frogsDataset import FrogsDataset as Dataset


def extractRepresentatives(samples, buckets, centroids, reprNum):
    res = {}  # {clusterNum : [representetives] }
    assert samples.ndim == 2

    for i in range(len(buckets.keys())):  # iterate buckets
        if len(buckets[i]) <= reprNum:
            res[i] = buckets[i].copy()
            continue

        sliced = samples[buckets[i], :]  # sliced = samples from bucket #(i)
        neighAlg = NearestNeighbors(n_neighbors=reprNum)
        neighAlg.fit(sliced)

        centerOfBucket = np.expand_dims(centroids[i], axis=0)
        neighInds = np.squeeze(neighAlg.kneighbors(centerOfBucket, return_distance=False),
                               axis=0)  # returns the indices in buckets[i] that correspondes to the k-nearest
        res[i] = [buckets[i][j] for j in neighInds]  # buckets[i][j] = the index of the sample in "samples".

    with open(settings.representativesPath, 'wb') as f:
        pcklr = pickle.Pickler(f)
        pcklr.dump(res)


def main():
    settings.sysAsserts()
    settings.representativesAsserts()

    dataset = Dataset(settings.frogs, settings.frogsSubset1C)
    dloader = DataLoader(dataset, batch_size=hyperparams.archMainBatchSize, shuffle=False)
    lowDimMat = reduceDim(dloader, settings.pcaPath)

    with open(settings.clusteringPath, 'rb') as f:
        pcklr = pickle.Unpickler(f)
        buckets, centroids = pcklr.load()

    extractRepresentatives(lowDimMat, buckets, centroids, hyperparams.clusteringReprNum)


if __name__ == '__main__':
    main()
