from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from src import hyperparameters as hyperparams
from src import settings
from src.clustering.pca import reduceDim
from src.frogsDataset import FrogsDataset as Dataset
from src.utils import storePickle


def createBuckets(numClusters, labels):
    '''
    returns: dict of (cluster_ind, [list of indices of pictures in this cluster])
    each index equals *the position in the dataset's array*, and not the index in filesystem
    '''

    samplesLen = len(labels)
    res_dict = {i: [] for i in range(numClusters)}
    # Iterate Samples and add the index of the sample to its corresponding cluster
    for sampleIndex in range(samplesLen):
        sampleLabel = labels[sampleIndex]
        res_dict[sampleLabel].append(sampleIndex)
    return res_dict


def main():
    settings.sysAsserts()
    settings.clusteringAsserts()

    dataset = Dataset(settings.frogs, settings.frogs6000)
    dloader = DataLoader(dataset, batch_size=settings.bigBatchSize, shuffle=False)
    lowDimMat = reduceDim(dloader, settings.pcaPath)

    kmeans = KMeans(n_clusters=hyperparams.clusteringNum, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                    precompute_distances=True, algorithm='elkan').fit(lowDimMat)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    buckets = createBuckets(hyperparams.clusteringNum, labels)
    for v in buckets.values():
        assert len(v) >= 2  # there must be at least two samples in each bucket

    storePickle(settings.clusteringPath, (buckets, centroids))


if __name__ == '__main__':
    main()
