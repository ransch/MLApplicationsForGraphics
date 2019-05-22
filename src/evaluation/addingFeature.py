import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.encoder import Encoder
from src.networks.generator import Generator

inds = [(2087, 2144, 2162), (1238, 1779, 1780)]


def genImages(images, ind, inda, indb):
    figpath = settings.featuresPath / f'{ind}+{indb}-{inda}.jpg'
    assert not figpath.is_file()
    grid = vutils.make_grid(images.cpu())
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig(figpath, dpi=600)
    # plt.show()


def main():
    settings.sysAsserts()
    settings.featuresFilesAsserts()
    dataset = Dataset(settings.frogs, settings.frogs3000)

    enc = Encoder().to(settings.device)
    gen = Generator().to(settings.device)
    enc.load_state_dict(torch.load(settings.archEncPath))
    gen.load_state_dict(torch.load(settings.archGenPath))
    enc.eval()
    gen.eval()

    with torch.no_grad():
        for ind, inda, indb in inds:
            img = dataset[ind - settings.frogs3000[0]]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
            imga = dataset[inda - settings.frogs3000[0]]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
            imgb = dataset[indb - settings.frogs3000[0]]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
            latent = enc(img)[0]
            latenta = enc(imga)[0]
            latentb = enc(imgb)[0]
            delta = latentb - latenta

            vectors = [latent, latent + delta]
            batch = torch.stack(vectors)
            images = gen(batch.view(len(batch), hyperparams.latentDim, 1, 1))

            genImages(images, ind, inda, indb)


if __name__ == '__main__':
    main()
