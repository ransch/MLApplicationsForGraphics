import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from src import settings

clusters = [117, 116, 108, 105, 102]
numEach = 5


def main():
    with open(settings.clusteringPath, 'rb') as f:
        pcklr = pickle.Unpickler(f)
        buckets, _ = pcklr.load()

    for k, v in buckets.items():
        print(f'k={k}: size={len(v)} --------------------------------------')
        print(v)
        print('\n\n')

    fig, axs = plt.subplots(len(clusters), numEach)
    for cluster_ind in range(len(clusters)):
        for i in range(numEach):
            cluster = clusters[cluster_ind]
            ind = buckets[cluster][i]
            img = mpimg.imread(str(settings.frogs / f'frog-{ind}.png'))
            axs[cluster_ind, i].imshow(img)

    plt.show()


if __name__ == '__main__':
    main()
