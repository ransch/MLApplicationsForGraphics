import datetime
# import winsound

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
from src.utils import L1L2Criterion, saveHyperParams, addNoise


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


def remaining_time(start_time, sofar, dsizeMain, dsizeSubset, mainBatchSize, subsetBatchSize, epochsNum, evalEvery):
    return (time.time() - start_time) / sofar * (
            (dsizeMain + (int(dsizeMain / mainBatchSize) + 1) * subsetBatchSize)
            * epochsNum + (dsizeMain + dsizeSubset) * int((epochsNum + 1) / evalEvery) - sofar)


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

            loss = genCriterion(subsetImages,  # loss2 on page
                                gen(enc(subsetImages).view(len(subsetImages), hyperparams.latentDim, 1, 1))).pow_(
                hyperparams.archSubsetLossPow).mul_(hyperparams.archSubsetLossGamma)

            if mode == 0:  # when training encoder (adding loss1 on page)
                loss.add_(encCriterion(embed(subsetInds), enc(subsetImages)).pow_(hyperparams.archSubsetLossPow).mul_(
                    hyperparams.archSubsetLossBeta))

            latVec = enc(images).view(len(images), hyperparams.latentDim, 1, 1)  # encoded images
            synthImages = gen(latVec)  # decoder(encoder(x))
            loss.add_(archCriterion(images, synthImages).pow_(  # adding loss3 on page
                hyperparams.archMainLossPow).mul_(hyperparams.archLossAlpha))

            noisedLatVec = addNoise(latVec, hyperparams.archPertMean, hyperparams.archPertStd)
            synthNoisedImages = gen(noisedLatVec)
            loss.add_(archCriterion(synthImages, synthNoisedImages).pow_(
                hyperparams.archPertPow).mul_(hyperparams.archPertCoeff))

            loss.backward()
            if mode == 0:
                encOptim.step()
            else:
                genOptim.step()

            sofar += len(images) + len(subsetImages)
            if sofar >= printevery:
                progressCallback(sofar, remaining_time(start_time, sofar, dsizeMain, dsizeSubset,
                                                       hyperparams.archMainBatchSize, hyperparams.archSubsetBatchSize,
                                                       epochsNum, evalEvery))
                printevery += settings.printevery

        if (epoch - 1) % evalEvery == 0:
            enc.eval()
            gen.eval()
            evalEveryCallback()
            total_loss_arch, processed1 = totalLossMain(enc, gen, dloaderMain, dsizeMain, archCriterion)
            total_loss_enc, total_loss_gen, processed2 = totalLossSubset(enc, gen, embed, dloaderSubset, dsizeSubset,
                                                                         encCriterion, genCriterion)

            weighted_loss = (total_loss_arch ** hyperparams.archMainLossPow) * hyperparams.archLossAlpha + \
                            (total_loss_gen ** hyperparams.archSubsetLossPow) * hyperparams.archSubsetLossGamma + (
                                    total_loss_enc ** hyperparams.archSubsetLossPow) * hyperparams.archSubsetLossBeta
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
    datasetSubset = Dataset(settings.frogs, settings.frogsSubset)
    datasetMain = Dataset(settings.frogs, settings.frogsMain)
    dsizeSubset = len(datasetSubset)
    dsizeMain = len(datasetMain)

    enc = Encoder().to(settings.device)
    gen = Generator().to(settings.device)
    embed = nn.Embedding(dsizeSubset, hyperparams.latentDim).to(settings.device)

    enc.load_state_dict(torch.load(settings.encModelPath))
    gen.load_state_dict(torch.load(settings.gloGenPath))
    embed.load_state_dict(torch.load(settings.gloLatentPath))

    dloaderSubset = DataLoader(datasetSubset, batch_size=hyperparams.archSubsetBatchSize, shuffle=False)
    dloaderMain = DataLoader(datasetMain, batch_size=hyperparams.archMainBatchSize, shuffle=False)
    encOptim = optim.Adam(enc.parameters(), lr=hyperparams.archEncAdamLr, betas=hyperparams.archEncAdamBetas)
    genOptim = optim.Adam(gen.parameters(), lr=hyperparams.archGenAdamLr, betas=hyperparams.archGenAdamBetas)
    percLoss = VGGDistance(hyperparams.archPercLossAlpha, hyperparams.archPercLossBeta, hyperparams.archLossPowAlpha,
                           hyperparams.archLossPowBeta).to(settings.device)
    l1l2Loss = L1L2Criterion(hyperparams.archL1L2LossAlpha, hyperparams.archL1L2LossBeta)

    totalParams = sum(p.numel() for p in enc.parameters() if p.requires_grad) + \
                  sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(str(datetime.datetime.now()))
    print(f'Training {totalParams} parameters')

    try:
        train(enc, gen, embed, dloaderSubset, dloaderMain, dsizeSubset, dsizeMain, l1l2Loss, percLoss, percLoss,
              encOptim, genOptim, hyperparams.archEpochsNum, hyperparams.archRatio, hyperparams.archEvalEvery,
              epochCallback, progressCallback, evalEveryCallback, archLossCallback, betterCallback, archEndCallback)
        # winsound.Beep(640, 1000)
        saveHyperParams(settings.archHyperPath)

    except Exception as e:
        print('An error occurred :(')
        print(e)
        # winsound.Beep(420, 1000)


if __name__ == '__main__':
    main()
