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
testFrogs1000 = list(range(6001, 7001))

printevery = 1000
samplesLen = 5

gloGenPath = localModels / 'glo/gen.pt'
gloLatentPath = localModels / 'glo/latent.pt'
gloVisPath = localModels / 'glo/glo.jpg'
gloProgressPath = localModels / 'glo/progress'
gloHyperPath = localModels / 'glo/hyperparams.py'

encModelPath = matureModels / 'arch/enc6.pt'
encVisPath = localModels / 'enc/enc.jpg'
encHyperPath = localModels / 'enc/hyperparams.py'

archEncPath = localModels / 'arch/enc.pt'
archGenPath = localModels / 'arch/gen.pt'
archVisPath = localModels / 'arch/arch.jpg'
archProgressPath = localModels / 'arch/progress'
archHyperPath = localModels / 'arch/hyperparams.py'

interPath = localModels / 'arch/arch-inter'
featuresPath = localModels / 'arch/arch-features'

clusteringBatchSize = 2000
clusteringPath = p / 'clustering/dim50-128/128.pkl'
representativesPath = p / 'clustering/dim50-128/128repr.pkl'
pcaPath = p / 'clustering/pca-dim2.pkl'


def sysAsserts():
    assert torch.backends.mkl.is_available()
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled


def gloFilesAsserts():
    assert len({gloGenPath.parent, gloLatentPath.parent, gloVisPath.parent, gloHyperPath.parent}) == 1
    if not gloGenPath.parent.is_dir():
        os.makedirs(gloGenPath.parent)
    if not gloProgressPath.is_dir():
        os.makedirs(gloProgressPath.parent)

    assert not gloGenPath.is_file()
    assert not gloLatentPath.is_file()
    assert not gloVisPath.is_file()
    assert not gloHyperPath.is_file()
    assert len(os.listdir(gloProgressPath)) == 0


def encFilesAsserts():
    assert len({encModelPath.parent, encVisPath.parent, encHyperPath.parent}) == 1
    if not encModelPath.parent.is_dir():
        os.makedirs(encModelPath.parent)

    assert not encModelPath.is_file()
    assert not encVisPath.is_file()
    assert not encHyperPath.is_file()
    assert gloLatentPath.is_file()


def archFilesAsserts():
    assert len({archEncPath.parent, archGenPath.parent, archVisPath.parent, archHyperPath.parent}) == 1
    if not archEncPath.parent.is_dir():
        os.makedirs(archEncPath.parent)
    if not archProgressPath.is_dir():
        os.makedirs(archProgressPath.parent)

    assert not archEncPath.is_file()
    assert not archGenPath.is_file()
    assert not archVisPath.is_file()
    assert not archHyperPath.is_file()
    assert len(os.listdir(archProgressPath)) == 0
    assert gloGenPath.is_file()
    assert gloLatentPath.is_file()
    assert encModelPath.is_file()


def interFilesAsserts():
    if not interPath.is_dir():
        os.makedirs(interPath.parent)

    assert archEncPath.is_file()
    assert archGenPath.is_file()


def featuresFilesAsserts():
    if not featuresPath.is_dir():
        os.makedirs(featuresPath.parent)

    assert archEncPath.is_file()
    assert archGenPath.is_file()


def pcaAsserts():
    if not pcaPath.parent.is_dir():
        os.makedirs(pcaPath.parent)
    assert not pcaPath.is_file()


def clusteringAsserts():
    if not clusteringPath.parent.is_dir():
        os.makedirs(clusteringPath.parent)
    assert not clusteringPath.is_file()
    assert pcaPath.is_file()


def representativesAsserts():
    if not representativesPath.parent.is_dir():
        os.makedirs(representativesPath.parent)
    assert not representativesPath.is_file()
    assert clusteringPath.is_file()
    assert pcaPath.is_file()
