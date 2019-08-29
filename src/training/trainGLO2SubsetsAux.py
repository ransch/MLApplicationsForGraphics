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


def lossCallback(subsetLoss, mainLoss, weightedLoss):
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


def endCallback(figpath, gloTrainingTimePath, epochs, evalEvery, elapsed_time):
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


def totalLoss(gen, embed, dloader, dsize, criterion):
    loss = .0
    processed = 0

    with torch.no_grad():
        for batch in dloader:
            inds = batch['ind'].to(settings.device).view(-1)
            images = batch['image'].to(device=settings.device, dtype=torch.float32)

            processed += len(images)
            loss += criterion(images,
                              gen(embed(inds).view(len(images), hyperparams.latentDim, 1, 1))).item() * images.size(0)
        loss /= dsize
    return loss, processed


def remaining_time(start_time, sofar, dsizeMain, dsizeSubset, mainBatchSize, subsetBatchSize, epochsNum, evalEvery):
    return (time.time() - start_time) / sofar * (
            (dsizeMain + (int(dsizeMain / mainBatchSize) + 1) * subsetBatchSize) * epochsNum
            + (dsizeMain + dsizeSubset) * int((epochsNum + 1) / evalEvery) - sofar)
