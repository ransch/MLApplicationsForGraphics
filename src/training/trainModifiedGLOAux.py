import torch

from src import hyperparameters as hyperparams
from src import settings


def updateBestPosneg(bestPosneg, posneg, embed):
    mat = embed.weight.data
    for ind in range(bestPosneg.shape[0]):
        lat = embed(torch.tensor([ind], device=settings.device)).view(-1)

        positives = torch.index_select(mat, 0, torch.tensor(posneg[ind][0], dtype=torch.int64,
                                                            device=settings.device)).sub_(lat).pow_(2).sum(dim=1)
        negatives = torch.index_select(mat, 0, torch.tensor(posneg[ind][1], dtype=torch.int64,
                                                            device=settings.device)).sub_(lat).pow_(2).sum(dim=1)

        pos = torch.argmax(positives)
        neg = torch.argmin(negatives)
        bestPosneg[ind, 0] = pos
        bestPosneg[ind, 1] = neg


def collect(embed, posneg, inds):
    selected = torch.index_select(posneg, 0, inds)
    pos = embed(selected[:, 0].view(-1))
    neg = embed(selected[:, 1].view(-1))
    return pos, neg


def totalLoss(gen, embed, bestPosneg, dloader, dsize, criterion):
    loss = .0
    processed = 0

    with torch.no_grad():
        for batch in dloader:
            inds = batch['ind'].to(settings.device).view(-1)
            images = batch['image'].to(device=settings.device, dtype=torch.float32)
            processed += len(images)

            lat = embed(inds)
            pos, neg = collect(embed, bestPosneg, inds)
            loss += criterion(images,
                              gen(lat.view(len(images), hyperparams.latentDim, 1, 1))).item() * images.size(0)
            term = lat.sub(pos).pow(2).sum(dim=1).sub_(lat.sub(neg).pow(2).sum(dim=1)) \
                       .add_(hyperparams.modifiedGLOThreshold).clamp_(min=0).mean().item() * images.size(0)
            loss += term * hyperparams.modifiedGLOTermCoeff
        loss /= dsize
    return loss, processed
