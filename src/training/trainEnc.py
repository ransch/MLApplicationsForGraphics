import datetime
# import winsound

import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src import hyperparameters as hyperparams
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.encoder import Encoder
from src.training.trainAux import *
from src.training.trainEncAux import *
from src.utils import L1L2Criterion, saveHyperParams


def train(embed, enc, dloader, dsize, criterion, optim, epochsNum, evalEvery, epochCallback, progressCallback,
          evalEveryCallback, lossCallback, betterCallback, endCallback):
    start_time = time.time()
    best_loss = math.inf
    sofar = 0
    printevery = settings.printevery

    embed.eval()
    enc.train()
    for epoch in range(1, epochsNum + 1):
        epochCallback(epochsNum, epoch)

        for batch in dloader:
            inds = batch['ind'].to(settings.device).view(-1)
            images = batch['image'].to(settings.device).type(torch.float32)

            optim.zero_grad()
            loss = criterion(embed(inds), enc(images))
            loss.backward()
            optim.step()

            sofar += len(inds)
            if sofar >= printevery:
                progressCallback(sofar, (time.time() - start_time) / sofar * (
                        dsize * epochsNum + dsize * int((epochsNum + 1) / evalEvery) - sofar))
                printevery += settings.printevery

        if (epoch - 1) % evalEvery == 0:
            enc.eval()
            evalEveryCallback()
            total_loss, processed = totalLoss(embed, enc, dloader, dsize, criterion)
            sofar += processed
            lossCallback(total_loss)
            if total_loss < best_loss:
                best_loss = total_loss
                betterCallback(enc)
            enc.train()

        printevery = sofar
    endCallback(str(settings.encVisPath), settings.encTrainingTimePath, epochsNum, evalEvery, time.time() - start_time)


def totalLoss(embed, enc, dloader, dsize, criterion):
    loss = .0
    processed = 0

    with torch.no_grad():
        for batch in dloader:
            inds = batch['ind'].to(settings.device).view(-1)
            images = batch['image'].to(settings.device).type(torch.float32)

            processed += len(images)
            loss += criterion(embed(inds), enc(images)).item() * images.size(0)
        loss /= dsize
    return loss, processed


def main():
    settings.sysAsserts()
    settings.encFilesAsserts()
    dataset = Dataset(settings.frogs, settings.frogsSubset1)
    dsize = len(dataset)

    enc = Encoder().to(settings.device)
    embed = nn.Embedding(len(dataset), hyperparams.latentDim).to(settings.device)
    embed.load_state_dict(torch.load(settings.gloLatentPath))

    dloader = DataLoader(dataset, batch_size=hyperparams.encBatchSize, shuffle=False)
    optimizer = optim.Adam(enc.parameters(), lr=hyperparams.encAdamLr, betas=hyperparams.encAdamBetas)
    criterion = L1L2Criterion(hyperparams.encLossAlpha, hyperparams.encLossBeta)

    print(str(datetime.datetime.now()))
    print(f'Training {sum(p.numel() for p in embed.parameters() if p.requires_grad)} parameters')

    try:
        train(embed, enc, dloader, dsize, criterion, optimizer, hyperparams.encEpochsNum, hyperparams.encEvalEvery,
              epochCallback, progressCallback, evalEveryCallback, lossCallback, betterCallback, endCallback)
        # winsound.Beep(640, 1000)

        saveHyperParams(settings.encHyperPath)


    except Exception as e:
        print('An error occurred :(')
        print(e)
        # winsound.Beep(420, 1000)


if __name__ == '__main__':
    main()
