import pickle

import matplotlib.pyplot as plt

from src import settings

with open(settings.pcaPath, 'rb') as f:
    pcklr = pickle.Unpickler(f)
    lowDimMat = pcklr.load()

with open(settings.clusteringPath, 'rb') as f:
    pcklr = pickle.Unpickler(f)
    buckets = pcklr.load()

labels = []

for i in range(lowDimMat.shape[0]):
    l = -1
    for k, v in buckets.items():
        if i in v:
            l = k
            break
    assert l >= 0
    labels.append(l)

x = lowDimMat[:, 0]
y = lowDimMat[:, 1]
plt.scatter(x, y, c=labels, alpha=0.75, marker='o')
plt.show()
