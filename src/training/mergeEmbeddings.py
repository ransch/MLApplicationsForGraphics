import torch
import torch.nn as nn

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset


def indMappings(datasetSubset, dataset):
    fileind_to_mainind = {dataset[i]['fileind'].item(): i for i in range(len(dataset))}
    return {i: fileind_to_mainind[datasetSubset[i]['fileind'].item()] for i in range(len(datasetSubset))}


def mergeEmbeddings(dataset1, dataset2, dataset, embed1Path, embed2Path, resPath):
    dsize1 = len(dataset1)
    dsize2 = len(dataset2)
    dsize = len(dataset)
    assert dsize == dsize1 + dsize2

    embed1 = nn.Embedding(dsize1, hyperparams.latentDim).to(settings.device)
    embed2 = nn.Embedding(dsize2, hyperparams.latentDim).to(settings.device)
    res = nn.Embedding(dsize, hyperparams.latentDim).to(settings.device)
    embed1.load_state_dict(torch.load(embed1Path))
    embed2.load_state_dict(torch.load(embed2Path))
    embed1.eval()
    embed2.eval()
    res.eval()

    ind1_to_ind = indMappings(dataset1, dataset)
    ind2_to_ind = indMappings(dataset2, dataset)
    matrix = torch.empty((dsize, hyperparams.latentDim))

    with torch.no_grad():
        for d, indMapping, embed in zip((dataset1, dataset2), (ind1_to_ind, ind2_to_ind), (embed1, embed2)):
            for i in range(len(d)):
                matrix[indMapping[i]] = embed(torch.tensor([i]).to(settings.device)).view(1, hyperparams.latentDim)

        res.weight.data.copy_(matrix)

    torch.save(res.state_dict(), resPath)


if __name__ == '__main__':
    dataset1 = Dataset(settings.frogs, settings.frogsSubset)
    dataset2 = Dataset(settings.frogs, settings.frogsMain)
    dataset = Dataset(settings.frogs, settings.frogs6000)
    embed1Path = settings.localModels / 'enc2/latent.pt'
    embed2Path = settings.localModels / 'glo3/latent.pt'
    resPath = settings.localModels / 'enc3/latent.pt'
    assert not resPath.is_file()
    mergeEmbeddings(dataset1, dataset2, dataset, embed1Path, embed2Path, resPath)
