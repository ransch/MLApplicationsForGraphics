import pickle

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.utils import save_image

from src import hyperparameters as hyperparams
from src import settings
from src.networks.generator import Generator

mean = 0
std = 1
cnt = 6000


def genImage(image, fileind):
    figpath = settings.synthPath / f'{fileind}-fake.jpg'
    assert not figpath.is_file()
    save_image(image, figpath)


def loadSampler(filepath):
    with open(filepath, 'rb') as f:
        pcklr = pickle.Unpickler(f)
        fit = pcklr.load()
    mean = torch.from_numpy(fit['mean']).to(device=settings.device)
    cov = torch.from_numpy(fit['cov']).to(device=settings.device)
    sampler = MultivariateNormal(mean, cov)
    return sampler


def main():
    settings.sysAsserts()
    settings.synthesizeFilesAsserts()

    gen = Generator().to(settings.device)
    gen.load_state_dict(torch.load(settings.localModels / 'glo5/gen.pt'))
    gen.eval()

    sampler = loadSampler(settings.localModels / 'glo5/gaussianFit.pkl')
    with torch.no_grad():
        for i in range(cnt):
            lat = sampler.sample()
            fake = gen(lat.view(1, hyperparams.latentDim, 1, 1).type(torch.float32))
            genImage(fake, i + 1)


if __name__ == '__main__':
    main()
