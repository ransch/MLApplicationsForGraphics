import pickle

import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset


def main():
    settings.sysAsserts()
    dataset = Dataset(settings.frogs, settings.frogs6000)
    dsize = len(dataset)

    embed = nn.Embedding(dsize, hyperparams.latentDim).to(settings.device)
    embed.load_state_dict(torch.load(settings.localModels / 'glo5/latent.pt'))
    dataMat = embed.weight.data.cpu().numpy()
    assert len(dataMat.shape) == 2 and dataMat.shape[0] == 6000 and dataMat.shape[1] == hyperparams.latentDim

    gmm = GaussianMixture(n_components=1, covariance_type='full', tol=1e-7, max_iter=100000, n_init=100)
    gmm.fit(dataMat)

    mean = gmm.means_
    cov = gmm.covariances_
    assert len(mean.shape) == 2 and mean.shape[0] == 1 and mean.shape[1] == hyperparams.latentDim
    assert len(cov.shape) == 3 and cov.shape[0] == 1 and cov.shape[1] == hyperparams.latentDim and cov.shape[
        2] == hyperparams.latentDim

    with open(settings.gaussianFitPath, 'wb') as f:
        pcklr = pickle.Pickler(f)
        pcklr.dump({'mean': mean, 'cov': cov})


if __name__ == '__main__':
    main()
