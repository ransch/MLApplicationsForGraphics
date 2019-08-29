import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils as vutils

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.generator import Generator
from src.utils import findOptimalLatentVector


def main():
    index = 18

    dataset = Dataset(settings.frogs, settings.frogs6000)
    gen = Generator().to(settings.device)
    embed = nn.Embedding(len(settings.frogs6000), hyperparams.latentDim).to(settings.device)
    gen.load_state_dict(torch.load(settings.matureModels / 'glototal-1000-epochs/gen.pt'))
    embed.load_state_dict(torch.load(settings.matureModels / 'glototal-1000-epochs/latent.pt'))
    gen.eval()
    embed.eval()

    image = dataset[index - 1]['image'].to(device=settings.device, dtype=torch.float32)
    latVec = findOptimalLatentVector(gen, image)
    optimalfake = gen(
        embed(torch.tensor([index - 1], device=settings.device)).view(1, hyperparams.latentDim, 1, 1)).squeeze_(0)
    fake = gen(latVec.view(1, hyperparams.latentDim, 1, 1)).squeeze_(0)

    with torch.no_grad():
        images = torch.stack([image, optimalfake, fake], 0)
        grid = vutils.make_grid(images.cpu())
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.show()


if __name__ == '__main__':
    main()
