import torch
from torchvision.utils import save_image

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.generator import Generator

mean = 0
std = 1
cnt = 20


def genImage(image, fileind):
    figpath = settings.synthPath / f'{fileind}-fake.jpg'
    assert not figpath.is_file()
    save_image(image, figpath)


def main():
    settings.sysAsserts()
    settings.synthesizeFilesAsserts()
    dataset = Dataset(settings.frogs, settings.frogs6000)
    dsize = len(dataset)

    gen = Generator().to(settings.device)

    gen.load_state_dict(torch.load(settings.matureModels / 'glo4 with noise/gen.pt'))
    gen.eval()

    with torch.no_grad():
        for i in range(cnt):
            latent = torch.empty(hyperparams.latentDim) \
                .to(settings.device).normal_(mean=mean, std=std).view(1, hyperparams.latentDim, 1, 1)
            fake = gen(latent)
            genImage(fake, i+1)


if __name__ == '__main__':
    main()
