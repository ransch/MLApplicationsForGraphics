import torch

from src import settings


def betterCallback(enc):
    torch.save(enc.state_dict(), settings.encModelPath)
    print('A better model has been found and has been serialized into fs')
