import os

import skimage.io
import torch
from skimage.color import rgba2rgb
from torch.utils.data import Dataset
from torchvision import transforms

class FrogsDataset(Dataset):
    def __init__(self, root, start_ind):
        self.root = root
        self.start_ind = start_ind
        self.trans = transforms.ToTensor()

    def __len__(self):
        ret = 0
        for file in os.listdir(self.root):
            if file.endswith(".png"):
                ret += 1
        return ret

    def __getitem__(self, idx):
        filename = f'frog-{idx + self.start_ind}.png'
        filepath = os.path.join(self.root, filename)
        img = rgba2rgb(skimage.io.imread(filepath))
        img = self.trans(img)
        sample = {'ind': torch.tensor([idx], dtype=torch.int64), 'image': img}
        return sample
