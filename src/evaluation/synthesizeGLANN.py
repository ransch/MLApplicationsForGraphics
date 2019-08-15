import torch

from src import hyperparameters as hyperparams
from src import settings
from src.evaluation.synthesize import genImage
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.generator import Generator
from src.networks.imleMapping import Mapping

mean = 0
std = 1
cnt = 20


def main():
    settings.sysAsserts()
    settings.synthesizeFilesAsserts()
    dataset = Dataset(settings.frogs, settings.frogs6000)
    dsize = len(dataset)

    mapping = Mapping().to(settings.device)
    gen = Generator().to(settings.device)

    mapping.load_state_dict(torch.load(settings.localModels / 'totalimle/mapping.pt'))
    gen.load_state_dict(torch.load(settings.localModels / 'glototal/gen.pt'))
    mapping.eval()
    gen.eval()

    with torch.no_grad():
        for i in range(cnt):
            noise = torch.empty(hyperparams.noiseDim).normal_(mean=0, std=1)
            lat = mapping(noise)
            fake = gen(lat.view(1, hyperparams.latentDim, 1, 1))
            genImage(fake, i + 1)


if __name__ == '__main__':
    main()
