import datetime
import math

import numpy as np
import torch.nn as nn
import torch.optim as optim

from src.networks.generator import Generator
from src.networks.imleMapping import Mapping
from src.training.trainAux import *
from src.training.trainIMLEAux import *
from src.utils import saveHyperParams, L2Criterion, findNearest


def train(mapping, gen, embed, subsetSize, batchSize, miniBatchSize, criterion, optimizer, epochsNum, itersNum,
          evalEvery, epochCallback, progressCallback, evalEveryCallback, lossCallback, betterCallback, endCallback):
    start_time = time.time()
    last_updated = start_time
    best_loss = math.inf

    mapping.train()
    embed.eval()
    gen.eval()

    for epoch in range(1, epochsNum + 1):
        epochCallback(epochsNum, epoch)

        noise = torch.empty(subsetSize, hyperparams.noiseDim, device=settings.device).normal_(mean=0, std=1)
        mappedNoise = mapping(noise)
        inds = torch.from_numpy(np.random.choice(embed.num_embeddings, batchSize, replace=False)) \
            .to(device=settings.device, dtype=torch.int64)
        lats = embed(inds)
        nearest = findNearest(mappedNoise, lats)

        for iteration in range(1, itersNum + 1):
            minibatch_inds_loc = torch.from_numpy(np.random.choice(batchSize, miniBatchSize, replace=False)) \
                .to(device=settings.device, dtype=torch.int64)
            minibatch_lats = torch.index_select(lats, 0, minibatch_inds_loc)
            minibatch_nearest = torch.index_select(nearest, 0, minibatch_inds_loc)
            minibatch_nearest_noise = torch.index_select(noise, 0, minibatch_nearest)

            optimizer.zero_grad()
            loss = criterion(minibatch_lats, minibatch_nearest_noise)
            loss.backward(retain_graph=True)
            optimizer.step()

        if (epoch - 1) % settings.imleprintevery == 0:
            progressCallback((time.time() - start_time) / epoch * (epochsNum - epoch))

        if (epoch - 1) % evalEvery == 0:
            mapping.eval()
            evalEveryCallback()
            total_loss = totalLoss(mapping, embed, subsetSize, criterion)
            lossCallback(total_loss)
            if total_loss < best_loss:
                best_loss = total_loss
                last_updated = time.time()
                betterCallback(epoch, mapping, gen)
            mapping.train()

    endCallback(str(settings.imleVisPath), settings.imleTrainingTimePath, epochsNum, evalEvery,
                last_updated - start_time)


def totalLoss(mapping, embed, subsetSize, criterion):
    loss = .0

    with torch.no_grad():
        noise = torch.empty(subsetSize, hyperparams.noiseDim, device=settings.device).normal_(mean=0, std=1)
        mappedNoise = mapping(noise)
        lats = embed.weight.data
        nearest = findNearest(mappedNoise, lats)
        nearest_noise = torch.index_select(noise, 0, nearest)

        loss = criterion(lats, nearest_noise)
    return loss.item()


def main():
    settings.sysAsserts()
    settings.imleFilesAsserts()

    mapping = Mapping().to(settings.device)
    gen = Generator().to(settings.device)
    embed = nn.Embedding(len(settings.frogs6000), hyperparams.latentDim).to(settings.device)

    gen.load_state_dict(torch.load(settings.localModels / 'glototal/gen.pt'))
    embed.load_state_dict(torch.load(settings.localModels / 'glototal/latent.pt'))

    optimizer = optim.Adam(mapping.parameters(), lr=hyperparams.imleAdamLr, betas=hyperparams.imleAdamBetas)
    criterion = L2Criterion()

    totalParams = sum(p.numel() for p in mapping.parameters() if p.requires_grad)
    print(str(datetime.datetime.now()))
    print(f'Training {totalParams} parameters')

    try:
        train(mapping, gen, embed, hyperparams.imleSubsetSize, hyperparams.imleBatchSize, hyperparams.imleMiniBatchSize,
              criterion, optimizer, hyperparams.imleEpochsNum, hyperparams.imleItersNum, hyperparams.imleEvalEvery,
              epochCallback, imleProgressCallback, evalEveryCallback, lossCallback, betterCallback, endCallback)
        saveHyperParams(settings.imleHyperPath)

    except Exception as e:
        print('An error occurred :(')
        print(e)


if __name__ == '__main__':
    main()
