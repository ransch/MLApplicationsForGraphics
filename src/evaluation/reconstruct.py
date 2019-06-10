import torch
import torch.nn as nn
from torchvision.utils import save_image

from src.frogsDataset import FrogsDataset as Dataset
from src import hyperparameters as hyperparams
from src import settings
from src.networks.generator import Generator
from src.networks.encoder import Encoder

num = 5
inds = settings.frogsSubset1[:num]


def genImage(image, fileind):
    figpath = settings.reconsPath / f'{fileind}-fake.jpg'
    assert not figpath.is_file()
    save_image(image, figpath)


def main():
    settings.sysAsserts()
    settings.reconstructFilesAsserts()
    dataset = Dataset(settings.frogs, settings.frogsSubset1)
    dsize = len(dataset)

    gen = Generator().to(settings.device)
    enc = Encoder().to(settings.device)
    # embed = nn.Embedding(6000, hyperparams.latentDim).to(settings.device)

    gen.load_state_dict(torch.load(settings.gloGenPath))
    enc.load_state_dict(torch.load(settings.encModelPath))
    # embed.load_state_dict(torch.load(settings.gloLatentPath))
    gen.eval()
    enc.eval()
    # embed.eval()

    with torch.no_grad():
        for ind in range(num):
            sample = dataset[ind]
            image = sample['image'].to(settings.device).unsqueeze_(0).type(torch.float32)
            fileind = sample['fileind'].item()
            # latent = embed(torch.tensor([ind - 1]).to(settings.device)).view(1, hyperparams.latentDim, 1, 1)
            latent = enc(image).view(1, hyperparams.latentDim, 1, 1)
            fake = gen(latent)
            genImage(fake, fileind)


if __name__ == '__main__':
    main()
