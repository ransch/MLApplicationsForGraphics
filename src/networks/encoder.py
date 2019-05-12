# Adapted from https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py

import torch.nn as nn

from src import hyperparameters as hyperparams, utils


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, hyperparams.encFeatureMapsSize, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hyperparams.encFeatureMapsSize) x 32 x 32
            nn.Conv2d(hyperparams.encFeatureMapsSize, hyperparams.encFeatureMapsSize * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hyperparams.encFeatureMapsSize * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hyperparams.encFeatureMapsSize*2) x 16 x 16
            nn.Conv2d(hyperparams.encFeatureMapsSize * 2, hyperparams.encFeatureMapsSize * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hyperparams.encFeatureMapsSize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hyperparams.encFeatureMapsSize*4) x 8 x 8
            nn.Conv2d(hyperparams.encFeatureMapsSize * 4, hyperparams.encFeatureMapsSize * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hyperparams.encFeatureMapsSize * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hyperparams.encFeatureMapsSize*8) x 4 x 4
            nn.Conv2d(hyperparams.encFeatureMapsSize * 8, hyperparams.latentDim, 4, 1, 0, bias=False),
            # state size. (hyperparams.latentDim*8) x 1 x 1
            utils.Flatten(),
            # state size. (hyperparams.latentDim*8)
            nn.Linear(hyperparams.latentDim, hyperparams.latentDim, bias=True)
            # state size. (hyperparams.latentDim*8)
        )

        self.main.apply(utils.weights_init)

    def forward(self, x):
        return self.main(x)
