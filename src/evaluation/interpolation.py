import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.encoder import Encoder
from src.networks.generator import Generator

num = 20
inds = [(1113, 1114), (1207, 1234), (1246, 1247), (1403, 1405), (1578, 1579), (1779, 1780), (2087, 2096), (2802, 2803)]


def genImages(images, inda, indb):
    figpath = settings.interPath / f'{inda}-{indb}.jpg'
    assert not figpath.is_file()
    grid = vutils.make_grid(images.cpu())
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig(figpath, dpi=600)
    # plt.show()


def main():
    settings.sysAsserts()
    settings.interFilesAsserts()
    dataset = Dataset(settings.frogs, settings.frogs3000)

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
        for inda, indb in inds:
            imga = dataset[inda - settings.frogs3000[0]]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
            imgb = dataset[indb - settings.frogs3000[0]]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
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

            genImages(images, inda, indb)


if __name__ == '__main__':
    main()
