import torch
import torch.nn as nn
from torchvision.utils import save_image

from src import hyperparameters as hyperparams
from src import settings
from src.networks.generator import Generator

inds = [168, 170, 226, 250, 273]


def genImage(image, ind):
    figpath = settings.reconsPath / f'{ind}-fake.jpg'
    assert not figpath.is_file()
    save_image(image, figpath)


def main():
    settings.sysAsserts()
    settings.reconstructFilesAsserts()

    gen = Generator().to(settings.device)
    embed = nn.Embedding(6000, hyperparams.latentDim).to(settings.device)
    gen.load_state_dict(torch.load(settings.gloGenPath))
    embed.load_state_dict(torch.load(settings.gloLatentPath))
    gen.eval()
    embed.eval()

    with torch.no_grad():
        for ind in inds:
            fake = gen(
                embed(torch.tensor([ind - 1]).to(settings.device)).view(1, hyperparams.latentDim, 1, 1))
            genImage(fake, ind)


if __name__ == '__main__':
    main()
