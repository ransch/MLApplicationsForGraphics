import datetime
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.generator import Generator
from src.perceptual_loss import VGGDistance
from src.training.trainAux import lossCallback, epochCallback, progressCallback, evalEveryCallback, endCallback
from src.training.trainGLOAux import betterCallback, totalLoss
from src.training.trainModifiedGLOAux import updateBestPosneg, collect
from src.utils import saveHyperParams, projectRowsToLpBall, loadPickle


def train(gen, embed, dloader, dsize, posneg, criterion, genOptim, embedOptim, epochsNum, evalEvery,
          computeEvery, epochCallback, progressCallback, evalEveryCallback, lossCallback, betterCallback, endCallback):
    start_time = time.time()
    last_updated = start_time
    best_loss = math.inf
    sofar = 0
    printevery = settings.printevery
    bestPosneg = torch.empty(dsize, 2, device=settings.device, dtype=torch.int64)  # [ind: (best pos, best neg)]

    gen.train()
    embed.train()
    for epoch in range(1, epochsNum + 1):
        epochCallback(epochsNum, epoch)

        if (epoch - 1) % computeEvery == 0:
            updateBestPosneg(bestPosneg, posneg, embed)

        for batch in dloader:
            inds = batch['ind'].to(settings.device).view(-1)
            images = batch['image'].to(device=settings.device, dtype=torch.float32)
            genOptim.zero_grad()
            embedOptim.zero_grad()

            lat = embed(inds)
            pos, neg = collect(embed, bestPosneg, inds)
            loss = criterion(images, gen(lat.view(len(images), hyperparams.latentDim, 1, 1)))

            term = lat.sub(pos).pow(2).sum(dim=1).mean().sub_(lat.sub(neg).pow(2).sum(dim=1).mean()) \
                .add_(hyperparams.modifiedGLOThreshold).clamp_(min=0)
            loss.add_(term.mul_(hyperparams.modifiedGLOTermCoeff))

            loss.backward()
            genOptim.step()
            embedOptim.step()
            with torch.no_grad():
                projectRowsToLpBall(embed.weight.data)

            sofar += len(images)
            if sofar >= printevery:
                progressCallback(sofar, (time.time() - start_time) / sofar * (
                        dsize * epochsNum + dsize * int((epochsNum + 1) / evalEvery) - sofar))
                printevery += settings.printevery

        if (epoch - 1) % evalEvery == 0:
            gen.eval()
            embed.eval()
            evalEveryCallback()
            total_loss, processed = totalLoss(gen, embed, dloader, dsize, criterion)
            sofar += processed
            lossCallback(total_loss)
            if total_loss < best_loss:
                best_loss = total_loss
                last_updated = time.time()
                betterCallback(epoch, gen, embed, dloader)
            gen.train()
            embed.train()

        printevery = sofar
    endCallback(str(settings.gloVisPath), settings.gloTrainingTimePath, epochsNum, evalEvery, last_updated - start_time)


def main():
    settings.sysAsserts()
    settings.gloFilesAsserts()
    dataset = Dataset(settings.frogs, settings.frogs6000)
    dsize = len(dataset)
    posneg = loadPickle(settings.p / 'clustering/6000-dim-100-clst-128/posneg.pkl')

    gen = Generator().to(settings.device)
    embed = nn.Embedding(dsize, hyperparams.latentDim).to(settings.device)
    projectRowsToLpBall(embed.weight.data)

    dloader = DataLoader(dataset, batch_size=hyperparams.gloBatchSize, shuffle=True)
    genOptim = optim.Adam(gen.parameters(), lr=hyperparams.genAdamLr, betas=hyperparams.genAdamBetas)
    embedOptim = optim.Adam(embed.parameters(), lr=hyperparams.embedAdamLr, betas=hyperparams.embedAdamBetas)
    criterion = VGGDistance(hyperparams.gloLossAlpha, hyperparams.gloLossBeta, hyperparams.gloLossPowAlpha,
                            hyperparams.gloLossPowBeta).to(settings.device)

    totalParams = sum(p.numel() for p in gen.parameters() if p.requires_grad) + \
                  sum(p.numel() for p in embed.parameters() if p.requires_grad)
    print(str(datetime.datetime.now()))
    print(f'Training {totalParams} parameters')

    try:
        train(gen, embed, dloader, dsize, posneg, criterion, genOptim, embedOptim, hyperparams.gloEpochsNum,
              hyperparams.gloEvalEvery, hyperparams.modifiedGLOComputeEvery, epochCallback, progressCallback,
              evalEveryCallback, lossCallback, betterCallback, endCallback)
        saveHyperParams(settings.gloHyperPath)

    except Exception as e:
        print('An error occurred :(')
        print(e)


if __name__ == '__main__':
    main()
