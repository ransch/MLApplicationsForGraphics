import numpy as np
import torch
import torch.nn as nn

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.utils import storePickle


def main():
    settings.sysAsserts()
    dataset = Dataset(settings.frogs, settings.frogs6000)
    dsize = len(dataset)

    embed = nn.Embedding(dsize, hyperparams.latentDim).to(settings.device)
    embed.load_state_dict(torch.load(settings.localModels / 'modifiedglo/latent.pt'))
    dataMat = embed.weight.data.cpu().numpy()
    assert len(dataMat.shape) == 2 and dataMat.shape[0] == 6000 and dataMat.shape[1] == hyperparams.latentDim

    mean = np.mean(dataMat, axis=0)
    cov = np.cov(dataMat, rowvar=False)

    assert len(mean.shape) == 1 and mean.shape[0] == hyperparams.latentDim
    assert len(cov.shape) == 2 and cov.shape[0] == hyperparams.latentDim and cov.shape[1] == hyperparams.latentDim

    storePickle(settings.gaussianFitPath, {'mean': mean, 'cov': cov})


if __name__ == '__main__':
    main()
