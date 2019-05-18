import datetime
import winsound

import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.frogsDataset import FrogsDataset as Dataset
from src.networks.encoder import Encoder
from src.networks.generator import Generator
from src.perceptual_loss import VGGDistance
from src.training.trainArchAux import *
from src.training.trainAux import *
from src.utils import L1L2PertCriterion


def setMode(epoch, ratio, enc, gen):
    if ((epoch - 1) % (ratio[0] + ratio[1]) + 1) <= ratio[0]:
        mode = 0  # enc
        enc.train()
        gen.eval()
    else:
        mode = 1  # gen
        enc.eval()
        gen.train()
    return mode


def train(enc, gen, embed, dloaderSubset, dloaderMain, dsizeSubset, dsizeMain, encCriterion, genCriterion,
          archCriterion, encOptim, genOptim, epochsNum, ratio, evalEvery, epochCallback, progressCallback,
          evalEveryCallback, lossCallback, betterCallback, endCallback):
    start_time = time.time()
    best_loss = math.inf
    sofar = 0
    printevery = settings.printevery

    embed.eval()
    for epoch in range(1, epochsNum + 1):
        epochCallback(epochsNum, epoch)
        mode = setMode(epoch, ratio, enc, gen)
        subsetIter = iter(dloaderSubset)

        for batch in dloaderMain:
            images = batch['image'].to(settings.device).type(torch.float32)
            try:
                subsetBatch = next(subsetIter)
            except StopIteration:
                subsetIter = iter(dloaderSubset)
                subsetBatch = next(subsetIter)
            subsetInds = subsetBatch['ind'].to(settings.device).view(-1)
            subsetImages = subsetBatch['image'].to(settings.device).type(torch.float32)

            if mode == 0:
                encOptim.zero_grad()
            else:
                genOptim.zero_grad()

            loss = genCriterion(subsetImages,
                                gen(enc(subsetImages).view(len(subsetImages), hyperparams.latentDim, 1, 1))).pow_(
                hyperparams.archSubsetLossPow).mul_(hyperparams.archSubsetLossGamma)

            if mode == 0:
                loss.add_(encCriterion(embed(subsetInds), enc(subsetImages)).pow_(hyperparams.archSubsetLossPow).mul_(
                    hyperparams.archSubsetLossBeta))

            loss.add_(archCriterion(images, gen(enc(images).view(len(images), hyperparams.latentDim, 1, 1))).pow_(
                hyperparams.archMainLossPow).mul_(hyperparams.archLossAlpha))
            loss.backward()
            if mode == 0:
                encOptim.step()
            else:
                genOptim.step()

            sofar += len(images) + len(subsetImages)
            if sofar >= printevery:
                progressCallback(sofar, (time.time() - start_time) / sofar * (
                        (dsizeMain + int(dsizeMain / hyperparams.archMainBatchSize * hyperparams.archSubsetBatchSize))
                        * epochsNum + (dsizeMain + dsizeSubset) * int((epochsNum + 1) / evalEvery) - sofar))
                printevery += settings.printevery

        if (epoch - 1) % evalEvery == 0:
            enc.eval()
            gen.eval()
            evalEveryCallback()
            total_loss_arch, processed1 = totalLossMain(enc, gen, dloaderMain, dsizeMain, archCriterion)
            total_loss_enc, total_loss_gen, processed2 = totalLossSubset(enc, gen, embed, dloaderSubset, dsizeSubset,
                                                                         encCriterion, genCriterion)

            weighted_loss = (total_loss_arch ** hyperparams.archMainLossPow) * hyperparams.archLossAlpha + \
                            (total_loss_gen ** hyperparams.archSubsetLossPow) * hyperparams.archSubsetLossGamma
            if mode == 0:
                weighted_loss += (total_loss_enc ** hyperparams.archSubsetLossPow) * hyperparams.archSubsetLossBeta
            sofar += processed1 + processed2

            lossCallback(total_loss_enc, total_loss_gen, total_loss_arch, weighted_loss)
            if total_loss_arch < best_loss:
                best_loss = total_loss_arch
                betterCallback(epoch, enc, gen, dloaderMain, dloaderSubset)

        printevery = sofar
    endCallback(str(settings.archVisPath), epochsNum, evalEvery, time.time() - start_time)


def totalLossMain(enc, gen, dloader, dsize, criterion):
    loss = .0
    processed = 0

    with torch.no_grad():
        for batch in dloader:
            images = batch['image'].to(settings.device).type(torch.float32)

            processed += len(images)
            loss += criterion(images,
                              gen(enc(images).view(len(images), hyperparams.latentDim, 1, 1))).item() * images.size(0)
        loss /= dsize
    return loss, processed


def totalLossSubset(enc, gen, embed, dloader, dsize, encCriterion, genCriterion):
    enc_loss = .0
    gen_loss = .0
    processed = 0

    with torch.no_grad():
        for batch in dloader:
            inds = batch['ind'].to(settings.device).view(-1)
            images = batch['image'].to(settings.device).type(torch.float32)
            processed += len(images)
            enc_loss += encCriterion(embed(inds), enc(images)).item() * images.size(0)
            gen_loss += genCriterion(images, gen(
                enc(images).view(len(images), hyperparams.latentDim, 1, 1))).item() * images.size(0)
        enc_loss /= dsize
        gen_loss /= dsize
    return enc_loss, gen_loss, processed


def main():
    settings.sysAsserts()
    settings.archFilesAsserts()
    datasetSubset = Dataset(settings.frogs, settings.frogs1000)
    datasetMain = Dataset(settings.frogs, settings.frogs3000)
    dsizeSubset = len(datasetSubset)
    dsizeMain = len(datasetMain)

    enc = Encoder().to(settings.device)
    gen = Generator().to(settings.device)
    embed = nn.Embedding(len(datasetSubset), hyperparams.latentDim).to(settings.device)

    enc.load_state_dict(torch.load(settings.encModelPath))
    gen.load_state_dict(torch.load(settings.gloGenPath))
    embed.load_state_dict(torch.load(settings.gloLatentPath))

    dloaderSubset = DataLoader(datasetSubset, batch_size=hyperparams.archSubsetBatchSize, shuffle=False)
    dloaderMain = DataLoader(datasetMain, batch_size=hyperparams.archMainBatchSize, shuffle=False)
    encOptim = optim.Adam(enc.parameters(), lr=hyperparams.archEncAdamLr, betas=hyperparams.archEncAdamBetas)
    genOptim = optim.Adam(gen.parameters(), lr=hyperparams.archGenAdamLr, betas=hyperparams.archGenAdamBetas)
    percLoss = VGGDistance(hyperparams.archPercLossAlpha, hyperparams.archPercLossBeta, hyperparams.archLossPowAlpha,
                           hyperparams.archLossPowBeta).to(settings.device)
    l1l2Loss = L1L2PertCriterion(hyperparams.archL1L2LossAlpha, hyperparams.archL1L2LossBeta,
                                 hyperparams.archL1L2PertMean, hyperparams.archL1L2PertStd,
                                 hyperparams.archL1L2PertAlpha, hyperparams.archL1L2PertBeta,
                                 hyperparams.archL1L2PertGamma)

    totalParams = sum(p.numel() for p in enc.parameters() if p.requires_grad) + \
                  sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(str(datetime.datetime.now()))
    print(f'Training {totalParams} parameters')

    try:
        train(enc, gen, embed, dloaderSubset, dloaderMain, dsizeSubset, dsizeMain, l1l2Loss, percLoss, percLoss,
              encOptim, genOptim, hyperparams.archEpochsNum, hyperparams.archRatio, hyperparams.archEvalEvery,
              epochCallback, progressCallback, evalEveryCallback, archLossCallback, betterCallback, archEndCallback)
        winsound.Beep(640, 1000)
    except:
        print('An error occurred :(')
        winsound.Beep(420, 1000)


if __name__ == '__main__':
    main()
