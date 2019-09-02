import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.utils import loadPickle


def loadImages(dataset, buckets):
    labels = [0 for i in range(len(dataset))]
    images = torch.empty(len(dataset), 3, 64, 64, device=settings.device)
    for i in range(len(dataset)):
        fileind = dataset[i]['fileind']
        images[i] = dataset[i]['image']

    for c in buckets.keys():
        for i in buckets[c]:
            labels[i] = c

    return labels, images


def main():
    settings.sysAsserts()
    writer = SummaryWriter(log_dir=settings.p / 'vis/app/public/modifiedglo')
    dataset = Dataset(settings.frogs, settings.frogs6000)
    buckets, _ = loadPickle(settings.p / 'clustering/6000-dim-100-clst-128/clusters.pkl')
    embed = nn.Embedding(len(dataset), hyperparams.latentDim).to(settings.device)
    embed.load_state_dict(torch.load(settings.localModels / 'modifiedglo/latent.pt'))
    embed.eval()

    labels, images = loadImages(dataset, buckets)

    writer.add_embedding(embed.weight.data, metadata=labels, label_img=images)
    writer.close()


if __name__ == '__main__':
    main()
