import datetime

import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.frogsDataset import FrogsDataset as Dataset
from src.networks.generator import Generator
from src.perceptual_loss import VGGDistance
from src.training.trainAux import *
from src.training.trainGLO2SubsetsAux import *
from src.utils import saveHyperParams


# import winsound

def remaining_time(start_time, sofar, dsizeMain, dsizeSubset, mainBatchSize, subsetBatchSize, epochsNum, evalEvery):
    return (time.time() - start_time) / sofar * (
            (dsizeMain + (int(dsizeMain / mainBatchSize) + 1) * subsetBatchSize) * epochsNum
            + (dsizeMain + dsizeSubset) * int((epochsNum + 1) / evalEvery) - sofar)


def train(gen, embed, embedSubset, dloaderMain, dloaderSubset, dsizeMain, dsizeSubset, criterion, genOptim, embedOptim,
          epochsNum, evalEvery, epochCallback, progressCallback, evalEveryCallback, lossCallback, betterCallback,
          endCallback):
    start_time = time.time()
    best_loss = math.inf
    sofar = 0
    printevery = settings.printevery

    gen.train()
    embed.train()
    embedSubset.eval()
    for epoch in range(1, epochsNum + 1):
        epochCallback(epochsNum, epoch)
        subsetIter = iter(dloaderSubset)

        for batch in dloaderMain:
            inds = batch['ind'].to(settings.device).view(-1)
            images = batch['image'].to(settings.device).type(torch.float32)
            try:
                subsetBatch = next(subsetIter)
            except StopIteration:
                subsetIter = iter(dloaderSubset)
                subsetBatch = next(subsetIter)
            subsetInds = subsetBatch['ind'].to(settings.device).view(-1)
            subsetImages = subsetBatch['image'].to(settings.device).type(torch.float32)

            genOptim.zero_grad()
            embedOptim.zero_grad()
            subsetLoss = criterion(subsetImages, gen(
                embedSubset(subsetInds).view(len(subsetImages), hyperparams.latentDim, 1, 1))).mul_(
                hyperparams.glo2SubsetCoeff)
            mainLoss = criterion(images, gen(embed(inds).view(len(images), hyperparams.latentDim, 1, 1))).mul_(
                hyperparams.glo2MainCoeff)

            loss = subsetLoss.add_(mainLoss)
            loss.backward()
            genOptim.step()
            embedOptim.step()

            sofar += len(images) + len(subsetImages)
            if sofar >= printevery:
                progressCallback(sofar, remaining_time(start_time, sofar, dsizeMain, dsizeSubset,
                                                       hyperparams.glo2MainBatchSize, hyperparams.glo2SubsetBatchSize,
                                                       epochsNum, evalEvery))
                printevery += settings.printevery

        if (epoch - 1) % evalEvery == 0:
            gen.eval()
            embed.eval()
            evalEveryCallback()
            total_loss_subset, processed1 = totalLoss(gen, embedSubset, dloaderSubset, dsizeSubset, criterion)
            total_loss_main, processed2 = totalLoss(gen, embed, dloaderMain, dsizeMain, criterion)
            sofar += processed1 + processed2
            weightedLoss = total_loss_subset * hyperparams.glo2SubsetCoeff + total_loss_main * hyperparams.glo2MainCoeff
            glo2LossCallback(total_loss_subset, total_loss_main, weightedLoss)
            if weightedLoss < best_loss:
                best_loss = weightedLoss
                betterCallback(epoch, gen, embed, embedSubset, dloaderMain, dloaderSubset)
            gen.train()
            embed.train()

        printevery = sofar
    glo2EndCallback(str(settings.gloVisPath), str(settings.gloTrainingTimePath), epochsNum, evalEvery,
                    time.time() - start_time)


def totalLoss(gen, embed, dloader, dsize, criterion):
    loss = .0
    processed = 0

    with torch.no_grad():
        for batch in dloader:
            inds = batch['ind'].to(settings.device).view(-1)
            images = batch['image'].to(settings.device).type(torch.float32)

            processed += len(images)
            loss += criterion(images,
                              gen(embed(inds).view(len(images), hyperparams.latentDim, 1, 1))).item() * images.size(0)
        loss /= dsize
    return loss, processed


def main():
    settings.sysAsserts()
    settings.gloFilesAsserts()
    datasetSubset = Dataset(settings.frogs, settings.frogsSubset1)
    datasetMain = Dataset(settings.frogs, settings.frogsSubset2)
    dsizeSubset = len(datasetSubset)
    dsizeMain = len(datasetMain)

    gen = Generator().to(settings.device)
    embedSubset = nn.Embedding(dsizeSubset, hyperparams.latentDim).to(settings.device)
    embed = nn.Embedding(dsizeMain, hyperparams.latentDim).to(settings.device)

    gen.load_state_dict(torch.load(settings.localModels / 'glo1/gen.pt'))
    embedSubset.load_state_dict(torch.load(settings.localModels / 'glo1/latent.pt'))

    dloaderSubset = DataLoader(datasetSubset, batch_size=hyperparams.glo2SubsetBatchSize, shuffle=False)
    dloaderMain = DataLoader(datasetMain, batch_size=hyperparams.glo2MainBatchSize, shuffle=False)
    genOptim = optim.Adam(gen.parameters(), lr=hyperparams.genAdamLr, betas=hyperparams.genAdamBetas)
    embedOptim = optim.Adam(embed.parameters(), lr=hyperparams.embedAdamLr, betas=hyperparams.embedAdamBetas)
    criterion = VGGDistance(hyperparams.gloLossAlpha, hyperparams.gloLossBeta, hyperparams.gloLossPowAlpha,
                            hyperparams.gloLossPowBeta).to(settings.device)

    totalParams = sum(p.numel() for p in gen.parameters() if p.requires_grad) + \
                  sum(p.numel() for p in embed.parameters() if p.requires_grad)
    print(str(datetime.datetime.now()))
    print(f'Training {totalParams} parameters')

    try:
        train(gen, embed, embedSubset, dloaderMain, dloaderSubset, dsizeMain, dsizeSubset, criterion, genOptim,
              embedOptim,
              hyperparams.gloEpochsNum, hyperparams.gloEvalEvery, epochCallback, progressCallback, evalEveryCallback,
              lossCallback, betterCallback,
              endCallback)
        # winsound.Beep(640, 1000)
        saveHyperParams(settings.gloHyperPath)

    except Exception as e:
        print('An error occurred :(')
        print(e)
        # winsound.Beep(420, 1000)


if __name__ == '__main__':
    main()
