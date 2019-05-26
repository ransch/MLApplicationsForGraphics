import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from src import hyperparameters as hyperparams
from src import settings


def createBuckets(numClusters, labels):
    res = {i: [] for i in range(numClusters)}
    for i in range(len(labels)):
        res[labels[i]].append(i)
    return res


def extractRepresentatives(matrix, buckets, centroids, reprNum):
    res = {}

    for i in range(len(buckets.keys())):
        if len(buckets[i]) <= 4:
            res[i] = buckets[i].copy()
            continue

        slicedLst = []
        for j in buckets[i]:
            slicedLst.append(matrix[j])

        sliced = np.vstack(slicedLst)
        kneighAlg = NearestNeighbors(n_neighbors=reprNum)
        kneighAlg.fit(sliced)
        kneighInds = np.squeeze(kneighAlg.kneighbors(np.expand_dims(centroids[i], axis=0))[1], axis=0)
        res[i] = [buckets[i][j] for j in kneighInds]

    return res


def main():
    settings.sysAsserts()
    settings.clusteringAsserts()

    with open(settings.pcaPath, 'rb') as f:
        pcklr = pickle.Unpickler(f)
        lowDimMat = pcklr.load()

    kmeans = KMeans(n_clusters=hyperparams.clusteringNum, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                    precompute_distances=True, algorithm='elkan').fit(lowDimMat)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    buckets = createBuckets(hyperparams.clusteringNum, labels)
    for v in buckets.values():
        assert len(v) > 0

    with open(settings.clusteringPath, 'wb') as f:
        pcklr = pickle.Pickler(f)
        pcklr.dump((buckets, centroids))


if __name__ == '__main__':
    main()
