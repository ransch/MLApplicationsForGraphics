import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.encoder import Encoder
from src.networks.generator import Generator

ind = 2021
inda = 2802
indb = 2803


def showImages(images):
    grid = vutils.make_grid(images.cpu())
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()


def main():
    settings.sysAsserts()
    settings.interFilesAsserts()
    dataset = Dataset(settings.frogs3000, settings.frogs3000Start)

    enc = Encoder().to(settings.device)
    gen = Generator().to(settings.device)
    enc.load_state_dict(torch.load(settings.encModelPath))
    gen.load_state_dict(torch.load(settings.gloGenPath))
    enc.eval()
    gen.eval()

    with torch.no_grad():
        img = dataset[ind - settings.frogs3000Start]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
        imga = dataset[inda - settings.frogs3000Start]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
        imgb = dataset[indb - settings.frogs3000Start]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
        latent = enc(img)[0]
        latenta = enc(imga)[0]
        latentb = enc(imgb)[0]
        delta = latentb - latenta

        vectors = [latent, latent + delta]
        batch = torch.stack(vectors)
        images = gen(batch.view(len(batch), hyperparams.latentDim, 1, 1))

        showImages(images)


if __name__ == '__main__':
    main()
