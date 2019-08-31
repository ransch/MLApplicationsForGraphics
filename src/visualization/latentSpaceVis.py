import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset


def loadImages(dataset):
    labels = []
    images = torch.empty(len(dataset), 3, 64, 64, device=settings.device)
    for i in range(len(dataset)):
        fileind = dataset[i]['fileind']
        labels.append(f'frog {fileind}')
        images[i] = dataset[i]['image']
    return labels, images


def main():
    settings.sysAsserts()
    writer = SummaryWriter(comment='glototal')
    dataset = Dataset(settings.frogs, settings.frogs6000)
    embed = nn.Embedding(len(dataset), hyperparams.latentDim).to(settings.device)
    embed.load_state_dict(torch.load(settings.matureModels / 'Z=l2 unit ball/glototal/latent.pt'))
    embed.eval()

    labels, images = loadImages(dataset)

    writer.add_embedding(embed.weight.data, metadata=labels, label_img=images)
    writer.close()


if __name__ == '__main__':
    main()
