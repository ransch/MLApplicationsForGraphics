import matplotlib.pyplot as plt
import torch
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
    gen.load_state_dict(torch.load(settings.gloGenPath))
    gen.eval()

    image = dataset[index - 1]['image'].to(settings.device).type(torch.float32)
    latVec = findOptimalLatentVector(gen, image)
    fake = gen(latVec.view(1, hyperparams.latentDim, 1, 1)).squeeze(0)

    with torch.no_grad():
        images = torch.stack([image, fake], 0)
        grid = vutils.make_grid(images.cpu())
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.show()


if __name__ == '__main__':
    main()
