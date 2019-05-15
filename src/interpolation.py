import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.encoder import Encoder
from src.networks.generator import Generator

inda = 2087
indb = 2096
num = 20


def genImages(images):
    # figpath = settings.interPath / f'{inda}-{indb}.jpg'
    # assert not figpath.is_file()
    grid = vutils.make_grid(images.cpu())
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    # plt.savefig(figpath, dpi=600)
    plt.show()


def main():
    settings.sysAsserts()
    settings.interFilesAsserts()
    dataset = Dataset(settings.frogs3000, settings.frogs3000Start)

    gen = Generator().to(settings.device)
    enc = Encoder().to(settings.device)
    # embed = nn.Embedding(len(dataset), hyperparams.latentDim).to(settings.device)
    gen.load_state_dict(torch.load(settings.archGenPath))
    enc.load_state_dict(torch.load(settings.archEncPath))
    # embed.load_state_dict(torch.load(settings.gloLatentPath))
    gen.eval()
    enc.eval()
    # embed.eval()

    with torch.no_grad():
        imga = dataset[inda - settings.frogs3000Start]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
        imgb = dataset[indb - settings.frogs3000Start]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
        latenta = enc(imga)[0]
        latentb = enc(imgb)[0]
        # latenta = embed(torch.tensor([inda - settings.frogs3000Start]).to(settings.device))[0]
        # latentb = embed(torch.tensor([indb - settings.frogs3000Start]).to(settings.device))[0]
        delta = (latentb - latenta) / (num + 1)

        vectors = [latenta]

        for i in range(num + 1):
            vectors.append(vectors[len(vectors) - 1] + delta)

        batch = torch.stack(vectors)
        images = gen(batch.view(len(batch), hyperparams.latentDim, 1, 1))

        genImages(images)


if __name__ == '__main__':
    main()
