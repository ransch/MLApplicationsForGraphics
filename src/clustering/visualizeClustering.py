import pickle

import matplotlib.pyplot as plt

from src import settings

with open(settings.pcaPath, 'rb') as f:
    pcklr = pickle.Unpickler(f)
    lowDimMat = pcklr.load()

with open(settings.clusteringPath, 'rb') as f:
    pcklr = pickle.Unpickler(f)
    buckets, centroids = pcklr.load()

with open(settings.representativesPath, 'rb') as f:
    pcklr = pickle.Unpickler(f)
    representatives = pcklr.load()

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
reprInds = []
for lst in representatives.values():
    reprInds.extend(lst)

reprMat = lowDimMat[reprInds, :]
reprX = reprMat[:, 0]
reprY = reprMat[:, 1]

plt.scatter(x, y, c=labels, alpha=0.75, marker='o')
plt.scatter(reprX, reprY, c="red", alpha=0.75, marker='X')
plt.show()
