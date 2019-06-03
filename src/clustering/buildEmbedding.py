import torch
import torch.nn as nn
from tqdm import tqdm

from src import hyperparameters as hyperparams
from src import settings
from src.frogsDataset import FrogsDataset as Dataset
from src.networks.encoder import Encoder


def main():
    encPath = settings.localModels / 'arch3/enc.pt'
    newEncPath = settings.localModels / 'latent.pt'

    dataset1 = Dataset(settings.frogs, settings.frogsSubset1)
    dataset2 = Dataset(settings.frogs, settings.frogsSubset2)
    dataset = Dataset(settings.frogs, settings.frogsSubset)

    enc = Encoder().to(settings.device)
    enc.load_state_dict(torch.load(encPath))
    embedding = nn.Embedding(len(dataset), hyperparams.latentDim).to(settings.device)
    enc.eval()
    embedding.eval()

    matrix = torch.empty((len(dataset), hyperparams.latentDim))

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            images = sample['image'].to(settings.device).type(torch.float32).unsqueeze_(0)
            matrix[i] = enc(images)

        embedding.weight.data.copy_(matrix)

    torch.save(embedding.state_dict(), newEncPath)


if __name__ == '__main__':
    main()
