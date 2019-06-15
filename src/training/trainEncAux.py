import os

import torch
from torchvision.utils import save_image

from src import hyperparameters as hyperparams
from src import settings


def betterCallback(epoch, gen, enc, dloader):
    torch.save(enc.state_dict(), settings.encModelPath)
    print('A better model has been found and has been serialized into fs')

    with torch.no_grad():
        batchiter = iter(dloader)
        batch = next(batchiter)
        fileinds = batch['fileind'].to(settings.device).view(-1)
        images = batch['image'].to(settings.device).type(torch.float32)
        assert len(fileinds) >= settings.samplesLen

        for i in range(settings.samplesLen):
            fileind = fileinds[i]
            image = images[i].unsqueeze_(0)
            fake = gen(enc(image).view(1, hyperparams.latentDim, 1, 1))
            filename = f'epoch-{epoch}-ind-{fileind.item()}.png'
            filepath = os.path.join(settings.encProgressPath, filename)
            save_image(fake[0], filepath)
