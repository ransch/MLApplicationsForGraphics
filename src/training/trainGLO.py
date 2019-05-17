import datetime
import winsound

import math
import torch.nn as nn
import torch.optim as optim
from scipy.fftpack import dst
from torch.utils.data import DataLoader

from src.frogsDataset import FrogsDataset as Dataset
from src.networks.generator import Generator
from src.perceptual_loss import VGGDistance
from src.training.trainAux import *
from src.training.trainGLOAux import *

import subprocess as sp


def train(gen, embed, dloader, dsize, criterion, genOptim, embedOptim, epochsNum, evalEvery, epochCallback,
          progressCallback, evalEveryCallback, lossCallback, betterCallback, endCallback):
    start_time = time.time()
    best_loss = math.inf
    sofar = 0
    printevery = settings.printevery

    gen.train()
    embed.train()
    for epoch in range(1, epochsNum + 1):
        epochCallback(epochsNum, epoch)

        for batch in dloader:
            inds = batch['ind'].to(settings.device).view(-1)
            images = batch['image'].to(settings.device).type(torch.float32)

            genOptim.zero_grad()
            embedOptim.zero_grad()
            loss = criterion(images, gen(embed(inds).view(len(images), hyperparams.latentDim, 1, 1)))
            loss.backward()
            genOptim.step()
            embedOptim.step()

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
                betterCallback(epoch, gen, embed, dloader)
            gen.train()
            embed.train()


        printevery = sofar
    endCallback(str(settings.gloVisPath), epochsNum, evalEvery, time.time() - start_time)
    ################################# YUVAL SERVER STUFF FOR THE NOOB ##########
    p = sp.Popen(args=["cp -R", settings.localModels, "../../../../../../cloudstorage/yuvalHelman/" ],
                 # COPY SOURCE DEST
                )
    p.wait()
    ################################# YUVAL SERVER STUFF FOR THE NOOB ##########


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
    # dataset = Dataset(settings.frogs, settings.frogs1000) # Yuval Edit
    dataset = Dataset(settings.frogs, settings.frogs1000)  # Yuval Edit TODO: change to 6k frogs
    dsize = len(dataset)

    gen = Generator().to(settings.device)
    embed = nn.Embedding(dsize, hyperparams.latentDim).to(settings.device)

    # gen.load_state_dict(torch.load(settings.p / 'models/glo-total/gen.pt'))
    # embed.load_state_dict(torch.load(settings.p / 'models/glo-total/latent.pt'))

    dloader = DataLoader(dataset, batch_size=hyperparams.gloBatchSize, shuffle=False)
    genOptim = optim.Adam(gen.parameters(), lr=hyperparams.genAdamLr, betas=hyperparams.genAdamBetas)
    embedOptim = optim.Adam(embed.parameters(), lr=hyperparams.embedAdamLr, betas=hyperparams.embedAdamBetas)
    criterion = VGGDistance(hyperparams.gloLossAlpha, hyperparams.gloLossBeta, hyperparams.gloLossPowAlpha,
                            hyperparams.gloLossPowBeta).to(settings.device)

    totalParams = sum(p.numel() for p in gen.parameters() if p.requires_grad) + \
                  sum(p.numel() for p in embed.parameters() if p.requires_grad)
    print(str(datetime.datetime.now()))
    print(f'Training {totalParams} parameters')

    try:
        train(gen, embed, dloader, dsize, criterion, genOptim, embedOptim, hyperparams.gloEpochsNum,
              hyperparams.gloEvalEvery, epochCallback, progressCallback, evalEveryCallback, lossCallback,
              betterCallback,
              endCallback)
        #winsound.Beep(640, 1000) # Yuval Edit

    except:
        print('An error occurred :(')
        #winsound.Beep(420, 1000) # Yuval Edit


if __name__ == '__main__':
    main()
