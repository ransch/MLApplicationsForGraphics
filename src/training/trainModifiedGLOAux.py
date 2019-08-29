import torch

from src import settings


def updatePosneg(posneg, buckets, lookup, embed):
    mat = embed.weight.data
    for ind in posneg.keys():
        lat = embed(torch.tensor([ind], device=settings.device)).view(-1)
        bucketInd, others = lookup[ind]
        sameBucket = buckets[bucketInd]
        sameBucket.remove(ind)

        positives = torch.index_select(mat, 0, torch.tensor(sameBucket, dtype=torch.int64, device=settings.device))
        positives.sub_(lat).pow_(2).sum(dim=1)
        negatives = torch.index_select(mat, 0, torch.tensor(others, dtype=torch.int64, device=settings.device))
        negatives.sub_(lat).pow_(2).sum(dim=1)

        try:
            pos = torch.argmax(positives)
            neg = torch.argmin(negatives)
        except:
            # todo!!!!!
            print("!!!")

        posneg[ind] = (pos, neg)


def collect(posneg, inds):
    pos = []
    neg = []

    for ind in inds:
        pos.append(posneg(ind)[0])
        neg.append(posneg(ind)[1])

    return torch.stack(pos, dim=0), torch.stack(neg, dim=0)
