import os
import time

import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image

from src import hyperparameters as hyperparams
from src import settings

_encLosses = []
_genLosses = []
_archLosses = []
_weightLosses = []


def archLossCallback(encLoss, genLoss, archLoss, weightedLoss):
    print(f'Training loss: {round(weightedLoss, 2)}')
    _encLosses.append(encLoss)
    _genLosses.append(genLoss)
    _archLosses.append(archLoss)
    _weightLosses.append(weightedLoss)


def betterCallback(epoch, enc, gen, dloaderMain, dloaderSubset):
    torch.save(enc.state_dict(), settings.archEncPath)
    torch.save(gen.state_dict(), settings.archGenPath)
    print('A better model has been found and has been serialized into fs')

    with torch.no_grad():
        for dlname, dloader in zip(('main', 'subset'), (dloaderMain, dloaderSubset)):
            batchiter = iter(dloader)
            next(batchiter)
            batch = next(batchiter)
            inds = batch['ind'].to(settings.device).view(-1)
            images = batch['image'].to(settings.device).type(torch.float32)
            assert len(inds) >= settings.samplesLen

            for i in range(settings.samplesLen):
                ind = inds[i]
                image = images[i].unsqueeze_(0)
                fake = gen(enc(image).view(1, hyperparams.latentDim, 1, 1))
                filename = f'epoch-{epoch}-{dlname}-ind-{ind.item()}.png'
                filepath = os.path.join(settings.archProgressPath, filename)
                save_image(fake[0], filepath)


def archEndCallback(figpath, epochs, evalEvery, elapsed_time):
    elapsed_time = time.gmtime(elapsed_time)
    print(f'Training finished in {elapsed_time.tm_mday - 1} days, '
          f'{elapsed_time.tm_hour} hours, {elapsed_time.tm_min} minutes')
    plt.plot(range(1, epochs + 1, evalEvery), _encLosses, '--o')
    plt.plot(range(1, epochs + 1, evalEvery), _genLosses, '--o')
    plt.plot(range(1, epochs + 1, evalEvery), _archLosses, '--o')
    # plt.plot(range(1, epochs + 1, evalEvery), _weightLosses, '--o')
    plt.legend(['Encoder loss', 'Generator loss', 'Arch loss'])  # , 'Weighted loss'])
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(figpath, dpi=600)
