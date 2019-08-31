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
imleprintevery = 5
samplesLen = 5

gloGenPath = localModels / 'modifiedglo/gen.pt'
gloLatentPath = localModels / 'modifiedglo/latent.pt'
gloVisPath = localModels / 'modifiedglo/vis.jpg'
gloProgressPath = localModels / 'modifiedglo/progress'
gloHyperPath = localModels / 'modifiedglo/hyperparams.py'
gloTrainingTimePath = localModels / 'modifiedglo/training_time.txt'

encModelPath = localModels / 'enc2/enc.pt'
encVisPath = localModels / 'enc2/vis.jpg'
encProgressPath = localModels / 'enc2/progress'
encHyperPath = localModels / 'enc2/hyperparams.py'
encTrainingTimePath = localModels / 'enc2/training_time.txt'

imleMappingPath = localModels / 'totalimle/mapping.pt'
imleVisPath = localModels / 'totalimle/vis.jpg'
imleProgressPath = localModels / 'totalimle/progress'
imleHyperPath = localModels / 'totalimle/hyperparams.py'
imleTrainingTimePath = localModels / 'totalimle/training_time.txt'

interPath = localModels / 'modifiedglo/eval/inter'
reconsPath = localModels / 'modifiedglo/eval/reconstruction'
synthPath = localModels / 'modifiedglo/eval/synth'

bigBatchSize = 2000
clusteringPath = p / 'clustering/6000-dim-100-clst-128/clusters.pkl'
representativesPath = p / 'clustering/6000-dim-100-clst-128/repr-8.pkl'
posnegPath = p / 'clustering/6000-dim-100-clst-128/posneg.pkl'
pcaPath = p / 'clustering/6000-pca-dim100.pkl'
gaussianFitPath = localModels / 'modifiedglo/gaussianFit.pkl'

frogs = p / 'frogs-64'
frogsAll = list(range(1, 7797))
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
assert len(set(frogs6000).intersection(set(testFrogs))) == 0
assert len(frogs6000) + len(testFrogs) == len(frogsAll)


def sysAsserts():
    assert torch.backends.mkl.is_available()
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled


def gloFilesAsserts():
    assert len({gloGenPath.parent, gloLatentPath.parent, gloVisPath.parent, gloHyperPath.parent,
                gloTrainingTimePath.parent}) == 1
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
    assert len({encModelPath.parent, encVisPath.parent, encHyperPath.parent, encTrainingTimePath.parent}) == 1
    if not encModelPath.parent.is_dir():
        os.makedirs(encModelPath.parent)
    if not encProgressPath.is_dir():
        os.makedirs(encProgressPath)

    assert not encModelPath.is_file()
    assert not encVisPath.is_file()
    assert not encHyperPath.is_file()
    assert not encTrainingTimePath.is_file()
    assert len(os.listdir(encProgressPath)) == 0


def imleFilesAsserts():
    assert len({imleMappingPath.parent, imleVisPath.parent, imleHyperPath.parent, imleTrainingTimePath.parent}) == 1
    if not imleMappingPath.parent.is_dir():
        os.makedirs(imleMappingPath.parent)
    if not imleProgressPath.is_dir():
        os.makedirs(imleProgressPath)

    assert not imleMappingPath.is_file()
    assert not imleVisPath.is_file()
    assert not imleHyperPath.is_file()
    assert not imleTrainingTimePath.is_file()
    assert len(os.listdir(imleProgressPath)) == 0


def interFilesAsserts():
    if not interPath.is_dir():
        os.makedirs(interPath)


def reconstructFilesAsserts():
    if not reconsPath.is_dir():
        os.makedirs(reconsPath)


def synthesizeFilesAsserts():
    if not synthPath.is_dir():
        os.makedirs(synthPath)


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


def posnegAsserts():
    if not posnegPath.parent.is_dir():
        os.makedirs(posnegPath.parent)
    assert not posnegPath.is_file()
