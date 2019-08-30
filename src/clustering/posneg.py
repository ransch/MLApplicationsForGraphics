from src import settings
from src.utils import loadPickle, storePickle


def genPosneg(indices, buckets):
    res = {}  # {ind : ([inds of other objects in the same cluster], [inds of objects in the other clusters])}

    for _, b_inds in buckets.items():
        neg = list(set(indices).difference(b_inds))

        for ind in b_inds:
            assert ind not in res
            pos = b_inds.copy()
            pos.remove(ind)
            assert len(pos) >= 1 and len(neg) >= 1
            res[ind] = (pos, neg)

    storePickle(settings.posnegPath, res)


def main():
    settings.sysAsserts()
    settings.posnegAsserts()

    indices = range(0, 6000)
    buckets, _ = loadPickle(settings.p / 'clustering/6000-dim-100-clst-128/clusters.pkl')
    genPosneg(indices, buckets)


if __name__ == '__main__':
    main()
