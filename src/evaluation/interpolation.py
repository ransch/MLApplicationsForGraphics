import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils as vutils

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.generator import Generator
from src.networks.encoder import Encoder

num = 20
inds = [(1113, 1114), (1207, 1234), (1246, 1247), (1568, 1574), (1779, 1780)]

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
    frogsInds = settings.frogs6000
    dataset = Dataset(settings.frogs, frogsInds)

    gen = Generator().to(settings.device)
    enc = Encoder().to(settings.device)
    embed = nn.Embedding(len(dataset), hyperparams.latentDim).to(settings.device)
    gen.load_state_dict(torch.load(settings.matureModels / 'glototal/gen.pt'))
    embed.load_state_dict(torch.load(settings.matureModels / 'glototal/latent.pt'))
    # enc.load_state_dict(torch.load(settings.encModelPath))
    gen.eval()
    embed.eval()
    enc.eval()

    for inda, indb in inds:
        imga = dataset[frogsInds.index(inda - 1)]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
        imgb = dataset[frogsInds.index(indb - 1)]['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
        # latenta = enc(imga)[0]
        # latentb = enc(imgb)[0]
        latenta = embed(torch.tensor([inda - 1]).to(settings.device)).view(hyperparams.latentDim, 1)
        latentb = embed(torch.tensor([indb - 1]).to(settings.device)).view(hyperparams.latentDim, 1)
        # latenta = findOptimalLatentVector(gen, imga.squeeze(0))
        # latentb = findOptimalLatentVector(gen, imgb.squeeze(0))
        delta = (latentb - latenta) / (num + 1)

        with torch.no_grad():
            vectors = [latenta]

            for i in range(num + 1):
                vectors.append(vectors[len(vectors) - 1] + delta)

            batch = torch.stack(vectors)
            images = gen(batch.view(len(batch), hyperparams.latentDim, 1, 1))

            genImages(images, inda, indb)


if __name__ == '__main__':
    main()
