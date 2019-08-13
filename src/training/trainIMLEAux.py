import os
import time

import torch
from torchvision.utils import save_image

from src import hyperparameters as hyperparams
from src import settings


def imleProgressCallback(remaining_time):
    remaining_time = time.gmtime(remaining_time)
    print(
        f'Remaining time: {remaining_time.tm_mday - 1} days, '
        f'{remaining_time.tm_hour} hours, {remaining_time.tm_min} minutes')


def betterCallback(epoch, mapping, gen):
    torch.save(mapping.state_dict(), settings.imleMappingPath)
    print('A better model has been found and has been serialized into fs')

    with torch.no_grad():
        for i in range(settings.samplesLen):
            noise = torch.empty(hyperparams.noiseDim).normal_(mean=0, std=1)
            lat = mapping(noise)
            fake = gen(lat.view(1, hyperparams.latentDim, 1, 1))
            filename = f'epoch-{epoch}-ind-{i}.png'
            filepath = os.path.join(settings.imleProgressPath, filename)
            save_image(fake[0], filepath)
