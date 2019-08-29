from src import settings
from src.utils import loadPickle

clusters = [81, 88, 102]
numEach = 5


def main():
    frogsInds = settings.frogs6000
    buckets, _ = loadPickle(settings.p / 'clustering/6000-dim-100-clst-128/clusters.pkl')

    for k, v in buckets.items():
        print(f'k={k}: size={len(v)} --------------------------------------')
        print(v)
        print('\n\n')


if __name__ == '__main__':
    main()
