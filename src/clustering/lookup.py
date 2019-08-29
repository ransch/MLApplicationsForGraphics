from src import settings
from src.utils import loadPickle, storePickle


def genLookup(indices, buckets):
    res = {}  # {ind : (bucketInd, [inds of other objects in the same cluster])}

    for cluster, b_inds in buckets.items():
        otherClusters = set(indices).difference(b_inds)
        for ind in b_inds:
            assert ind not in res
            res[ind] = (cluster, list(otherClusters.difference([ind])))

    storePickle(settings.lookupPath, res)


def main():
    settings.sysAsserts()
    settings.lookupAsserts()

    indices = range(0, 6000)
    buckets, _ = loadPickle(settings.p / 'clustering/6000-dim-100-clst-128/clusters.pkl')
    genLookup(indices, buckets)


if __name__ == '__main__':
    main()
