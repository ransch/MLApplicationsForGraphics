import os
import pickle
from pathlib import Path

import torch


def loadSubset(frogsInds, reprPath):
    with open(reprPath, 'rb') as f:
        pcklr = pickle.Unpickler(f)
        l = pcklr.load()
    return [frogsInds[item] for sublist in l.values() for item in sublist]


device = torch.device('cuda:0')

p = Path(__file__).resolve().parent.parent

localModels = p / 'no_git/models'
matureModels = p / 'models'

printevery = 1000
samplesLen = 5

gloGenPath = matureModels / 'glototal/gen.pt'
gloLatentPath = matureModels / 'glototal/latent.pt'
gloVisPath = matureModels / 'glototal/glo.jpg'
gloProgressPath = matureModels / 'glototal/progress'
gloHyperPath = matureModels / 'glototal/hyperparams.py'
gloTrainingTimePath = matureModels / 'glototal/training_time.txt'

encModelPath = matureModels / 'enc/enc.pt'
encVisPath = matureModels / 'enc/enc.jpg'
encProgressPath = matureModels / 'enc/progress'
encHyperPath = matureModels / 'enc/hyperparams.py'
encTrainingTimePath = matureModels / 'enc/training_time.txt'

archEncPath = localModels / 'arch/arch/enc.pt'
archGenPath = localModels / 'arch/gen.pt'
archVisPath = localModels / 'arch/arch.jpg'
archProgressPath = localModels / 'arch/progress'
archHyperPath = localModels / 'arch/hyperparams.py'
archTrainingTimePath = localModels / 'arch/training_time.txt'

interPath = matureModels / 'glo4 with noise/eval/inter'
featuresPath = localModels / 'arch/arch-features'
reconsPath = matureModels / 'glo4 with noise/eval/reconstruction'

clusteringBatchSize = 2000
clusteringPath = p / 'clustering/5488-dim-100-clst-128/clusters.pkl'
representativesPath = p / 'clustering/5488-dim-100-clst-128/repr-8.pkl'
pcaPath = p / 'clustering/pca-dim100.pkl'
clusteringVisPath = p / 'clustering/5488-dim-100-clst-128/vis.jpg'

frogs = p / 'frogs-64'
frogs1000 = list(range(1, 1001))
frogs3000 = list(range(1001, 4001))
frogs5000 = list(range(1001, 6001))
frogs6000 = frogs1000 + frogs5000
testFrogs = list(range(6001, 7796 + 1))
frogsSubset1 = loadSubset(frogs6000, p / 'clustering/6000-dim-100-clst-128/repr-4.pkl')
frogsSubset1C = sorted(set(frogs6000).difference(frogsSubset1))
frogsSubset2 = loadSubset(frogsSubset1C, p / 'clustering/5488-dim-100-clst-128/repr-8.pkl')
frogsSubset = sorted(set(frogsSubset1).union(frogsSubset2))
frogsMain = sorted(set(frogs6000).difference(frogsSubset))
assert len(set(frogsSubset1).intersection(set(frogsSubset2))) == 0
assert len(frogsMain) + len(frogsSubset1) + len(frogsSubset2) == 6000


def sysAsserts():
    assert torch.backends.mkl.is_available()
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled


def gloFilesAsserts():
    assert len({gloGenPath.parent, gloLatentPath.parent, gloVisPath.parent, gloHyperPath.parent}) == 1
    if not gloGenPath.parent.is_dir():
        os.makedirs(gloGenPath.parent)
    if not gloProgressPath.is_dir():
        os.makedirs(gloProgressPath)

    assert not gloGenPath.is_file()
    assert not gloLatentPath.is_file()
    assert not gloVisPath.is_file()
    assert not gloHyperPath.is_file()
    assert not gloTrainingTimePath.is_file()
    assert len(os.listdir(gloProgressPath)) == 0


def encFilesAsserts():
    assert len({encModelPath.parent, encVisPath.parent, encHyperPath.parent}) == 1
    if not encModelPath.parent.is_dir():
        os.makedirs(encModelPath.parent)
    if not encProgressPath.is_dir():
        os.makedirs(encProgressPath)

    assert not encModelPath.is_file()
    assert not encVisPath.is_file()
    assert not encHyperPath.is_file()
    assert not encTrainingTimePath.is_file()
    assert len(os.listdir(encProgressPath)) == 0


def archFilesAsserts():
    assert len({archEncPath.parent, archGenPath.parent, archVisPath.parent, archHyperPath.parent}) == 1
    if not archEncPath.parent.is_dir():
        os.makedirs(archEncPath.parent)
    if not archProgressPath.is_dir():
        os.makedirs(archProgressPath)

    assert not archEncPath.is_file()
    assert not archGenPath.is_file()
    assert not archVisPath.is_file()
    assert not archHyperPath.is_file()
    assert not archTrainingTimePath.is_file()
    assert len(os.listdir(archProgressPath)) == 0
    assert gloGenPath.is_file()
    assert gloLatentPath.is_file()
    assert encModelPath.is_file()


def interFilesAsserts():
    if not interPath.is_dir():
        os.makedirs(interPath)


def featuresFilesAsserts():
    if not featuresPath.is_dir():
        os.makedirs(featuresPath)

    assert archEncPath.is_file()
    assert archGenPath.is_file()


def reconstructFilesAsserts():
    if not reconsPath.is_dir():
        os.makedirs(reconsPath)


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
