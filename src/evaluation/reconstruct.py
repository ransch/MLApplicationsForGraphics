import torch
import torch.nn as nn
from torchvision.utils import save_image

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.encoder import Encoder
from src.networks.generator import Generator

inds = [166, 167, 169, 225, 249, 272]  # [settings.frogsMain[i] for i in range(10)]


def genImage(image, fileind):
    figpath = settings.reconsPath / f'{fileind}-fake.jpg'
    assert not figpath.is_file()
    save_image(image, figpath)


def main():
    settings.sysAsserts()
    settings.reconstructFilesAsserts()
    dataset = Dataset(settings.frogs, settings.frogs6000)
    dsize = len(dataset)

    gen = Generator().to(settings.device)
    enc = Encoder().to(settings.device)
    embed = nn.Embedding(dsize, hyperparams.latentDim).to(settings.device)

    gen.load_state_dict(torch.load(settings.matureModels / 'glo4 with noise/gen.pt'))
    enc.load_state_dict(torch.load(settings.matureModels / 'glo4 with noise/gen.pt'))
    # embed.load_state_dict(torch.load(settings.matureModels / 'glo4 with noise/latent.pt'))
    gen.eval()
    enc.eval()
    embed.eval()

    with torch.no_grad():
        for ind in inds:
            sample = dataset[ind]
            image = sample['image'].to(settings.device).unsqueeze_(0).type(torch.float32)
            fileind = sample['fileind'].item()
            latent = embed(torch.tensor([ind]).to(settings.device)).view(1, hyperparams.latentDim, 1, 1)
            # latent = enc(image).view(1, hyperparams.latentDim, 1, 1)
            fake = gen(latent)
            genImage(fake, fileind)


if __name__ == '__main__':
    main()
