# Adapted from https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py

import torch.nn as nn

from src import hyperparameters as hyperparams, utils


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(hyperparams.latentDim, hyperparams.genFeatureMapsSize * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hyperparams.genFeatureMapsSize * 8),
            nn.ReLU(True),
            # state size. (hyperparams.genFeatureMapsSize*8) x 4 x 4
            nn.ConvTranspose2d(hyperparams.genFeatureMapsSize * 8, hyperparams.genFeatureMapsSize * 4, 4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(hyperparams.genFeatureMapsSize * 4),
            nn.ReLU(True),
            # state size. (hyperparams.genFeatureMapsSize*4) x 8 x 8
            nn.ConvTranspose2d(hyperparams.genFeatureMapsSize * 4, hyperparams.genFeatureMapsSize * 2, 4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(hyperparams.genFeatureMapsSize * 2),
            nn.ReLU(True),
            # state size. (hyperparams.genFeatureMapsSize*2) x 16 x 16
            nn.ConvTranspose2d(hyperparams.genFeatureMapsSize * 2, hyperparams.genFeatureMapsSize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hyperparams.genFeatureMapsSize),
            nn.ReLU(True),
            # state size. (hyperparams.genFeatureMapsSize) x 32 x 32
            nn.ConvTranspose2d(hyperparams.genFeatureMapsSize, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )
        self.main.apply(utils.weights_init)

    def forward(self, x):
        return self.main(x)
