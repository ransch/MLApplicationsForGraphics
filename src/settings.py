import os
from pathlib import Path

import torch

device = torch.device('cuda:0')

p = Path('C:/Users/97250/PycharmProjects/MLApplicationsForGraphics')

frogs1000 = p / 'frogs-64/1000'
frogs3000 = p / 'frogs-64/3000'
frogs6796 = p / 'frogs-64/rest'
frogsall = p / 'frogs-64/all'
frogs1000Start = 1
frogs3000Start = 1001
frogs6796Start = 1001
frogsallStart = 1

printevery = 1000
samplesLen = 5

gloGenPath = p / 'models/glo-total/gen.pt'
gloLatentPath = p / 'models/glo-total/latent.pt'
gloVisPath = p / 'models/glo-total/glo.jpg'
gloProgressPath = p / 'models/glo-total/progress'

encModelPath = p / 'models/arch/enc2.pt'
encVisPath = p / 'models/enc/enc.jpg'

archEncPath = p / 'models/arch/enc3.pt'
archGenPath = p / 'models/arch/gen3.pt'
archVisPath = p / 'models/arch/arch3.jpg'
archProgressPath = p / 'models/arch/progress3'


def sysAsserts():
    assert torch.backends.mkl.is_available()
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled


def gloFilesAsserts():
    assert not gloGenPath.is_file()
    assert not gloLatentPath.is_file()
    assert not gloVisPath.is_file()
    assert gloProgressPath.is_dir()
    assert len(os.listdir(gloProgressPath)) == 0


def encFilesAsserts():
    assert not encModelPath.is_file()
    assert not encVisPath.is_file()
    assert gloLatentPath.is_file()


def archFilesAsserts():
    assert not archEncPath.is_file()
    assert not archGenPath.is_file()
    assert not archVisPath.is_file()
    assert archProgressPath.is_dir()
    assert len(os.listdir(archProgressPath)) == 0
    assert gloGenPath.is_file()
    assert gloLatentPath.is_file()
    assert encModelPath.is_file()


def interFilesAsserts():
    assert archEncPath.is_file()
    assert archGenPath.is_file()
    assert archVisPath.is_file()
