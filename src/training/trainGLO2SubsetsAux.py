import os
import time

import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image

from src import hyperparameters as hyperparams
from src import settings
from src.training.trainAux import printLogTrainingTime

_subsetLosses = []
_mainLosses = []
_weightedLosses = []


def glo2LossCallback(subsetLoss, mainLoss, weightedLoss):
    print(f'Training (weighted) loss: {round(weightedLoss, 2)}')
    _subsetLosses.append(subsetLoss)
    _mainLosses.append(mainLoss)
    _weightedLosses.append(weightedLoss)


def betterCallback(epoch, gen, embedMain, embedSubset, dloaderMain, dloaderSubset):
    torch.save(gen.state_dict(), settings.gloGenPath)
    torch.save(embedMain.state_dict(), settings.gloLatentPath)
    print('A better model has been found and has been serialized into fs')

    with torch.no_grad():
        for dlname, dloader, embed in zip(('main', 'subset'), (dloaderMain, dloaderSubset), (embedMain, embedSubset)):
            batchiter = iter(dloader)
            batch = next(batchiter)
            inds = batch['ind'].to(settings.device).view(-1)
            fileinds = batch['fileind'].to(settings.device).view(-1)
            assert len(fileinds) >= settings.samplesLen

            for i in range(settings.samplesLen):
                ind = inds[i]
                fileind = fileinds[i]
                fake = gen(embed(ind).view(1, hyperparams.latentDim, 1, 1))
                filename = f'epoch-{epoch}-{dlname}-ind-{fileind.item()}.png'
                filepath = os.path.join(settings.gloProgressPath, filename)
                save_image(fake[0], filepath)


def glo2EndCallback(figpath, gloTrainingTimePath, epochs, evalEvery, elapsed_time):
    elapsed_time = time.gmtime(elapsed_time)
    printLogTrainingTime(elapsed_time, gloTrainingTimePath)

    plt.plot(range(1, epochs + 1, evalEvery), _subsetLosses, '--o')
    plt.plot(range(1, epochs + 1, evalEvery), _mainLosses, '--o')
    # plt.plot(range(1, epochs + 1, evalEvery), _weightLosses, '--o')
    plt.legend(['Subset loss', 'Main loss'])  # , 'Weighted loss'])
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(figpath, dpi=600)
