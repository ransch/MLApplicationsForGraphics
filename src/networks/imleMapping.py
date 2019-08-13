import torch.nn as nn

from src import hyperparameters as hyperparams, utils


class Mapping(nn.Module):
    def __init__(self):
        super(Mapping, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(hyperparams.latentDim, hyperparams.latentDim),
            nn.Linear(hyperparams.latentDim, hyperparams.latentDim),
            nn.ReLU(True),
            nn.BatchNorm1d(hyperparams.latentDim)
        )
        self.main.apply(utils.weights_init)

    def forward(self, x):
        return self.main(x)
