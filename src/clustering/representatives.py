import pickle

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src import settings


def extractRepresentatives(samples, buckets, centroids, reprNum):
    res = {} #  {clusterNum : [representetives] }
    assert samples.ndim == 2

    for i in range(len(buckets.keys())): # iterate buckets
        if len(buckets[i]) <= reprNum:
            res[i] = buckets[i].copy()
            continue

        sliced = samples[buckets[i],:] # sliced = samples from bucket #(i)
        neighAlg = NearestNeighbors(n_neighbors=reprNum)
        neighAlg.fit(sliced)

        centerOfBucket = np.expand_dims(centroids[i], axis=0)
        neighInds = np.squeeze(neighAlg.kneighbors(centerOfBucket, return_distance=False), axis=0) # returns the indices in buckets[i] that correspondes to the k-nearest
        res[i] = [buckets[i][j] for j in neighInds] # buckets[i][j] = the index of the sample in "samples"

    with open(settings.representativesPath, 'wb') as f:
        pcklr = pickle.Pickler(f)
        pcklr.dump(res)


def main():
    settings.sysAsserts()
    settings.representativesAsserts()

    with open(settings.pcaPath, 'rb') as f:
        pcklr = pickle.Unpickler(f)
        lowDimMat = pcklr.load()

    with open(settings.clusteringPath, 'rb') as f:
        pcklr = pickle.Unpickler(f)
        buckets, centroids = pcklr.load()

    extractRepresentatives(lowDimMat, buckets, centroids, 1)


if __name__ == '__main__':
    main()
