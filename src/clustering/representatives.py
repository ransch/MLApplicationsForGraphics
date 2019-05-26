import pickle

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src import settings


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
