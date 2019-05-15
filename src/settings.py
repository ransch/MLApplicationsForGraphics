import os
from pathlib import Path

import torch

device = torch.device('cuda:0')

p = Path(__file__).resolve().parent.parent

localModels = p / 'no_git/models'
matureModels = p / 'models'

frogs = p / 'frogs-64'
frogs1000 = list(range(1, 1001))
frogs3000 = list(range(1001, 4001))
frogs5000 = list(range(1001, 6001))
frogs6000 = frogs1000 + frogs5000

printevery = 1000
samplesLen = 5

gloGenPath = localModels / 'arch/gen.pt'
gloLatentPath = matureModels / 'glo/latent.pt'
gloVisPath = localModels / 'glo/glo.jpg'
gloProgressPath = localModels / 'glo/progress'

encModelPath = localModels / 'arch/enc.pt'
encVisPath = localModels / 'enc/enc.jpg'

archEncPath = localModels / 'arch/enc2.pt'
archGenPath = localModels / 'arch/gen2.pt'
archVisPath = localModels / 'arch/arch2.jpg'
archProgressPath = localModels / 'arch/progress2'

interPath = localModels / 'arch/arch3-inter'

featuresPath = localModels / 'arch/arch3-features'


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
    assert interPath.is_dir()


def featuresFilesAsserts():
    assert archEncPath.is_file()
    assert archGenPath.is_file()
    assert featuresPath.is_dir()
