import os

import torch
from torchvision.utils import save_image

from src import hyperparameters as hyperparams
from src import settings


def betterCallback(epoch, gen, embed, dloader):
    torch.save(gen.state_dict(), settings.gloGenPath)
    torch.save(embed.state_dict(), settings.gloLatentPath)
    print('A better model has been found and has been serialized into fs')

    with torch.no_grad():
        batchiter = iter(dloader)
        next(batchiter)
        batch = next(batchiter)
        inds = batch['ind'].to(settings.device).view(-1)
        fileinds = batch['fileind'].to(settings.device).view(-1)
        assert len(inds) >= settings.samplesLen

        for i in range(settings.samplesLen):
            ind = inds[i]
            fileind = fileinds[i]
            fake = gen(embed(ind).view(1, hyperparams.latentDim, 1, 1))  # TODO
            filename = f'epoch-{epoch}-ind-{fileind.item()}.png'
            filepath = os.path.join(settings.gloProgressPath, filename)
            save_image(fake[0], filepath)
