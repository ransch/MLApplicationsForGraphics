import os

import skimage.io
import torch
from skimage.color import rgba2rgb
from torch.utils.data import Dataset
from torchvision import transforms


class FrogsDataset(Dataset):
    def __init__(self, root, indices):
        self.root = root
        self.indices = indices
        self.trans = transforms.ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        filename = f'frog-{self.indices[idx]}.png'
        filepath = os.path.join(self.root, filename)
        img = rgba2rgb(skimage.io.imread(filepath))
        img = self.trans(img)
        sample = {'ind': torch.tensor([idx], dtype=torch.int64), 'image': img}
        return sample
