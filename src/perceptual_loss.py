# Adapted from https://raw.githubusercontent.com/facebookresearch/NAM/master/code/perceptual_loss.py

import torch
import torch.nn as nn
import torchvision.models as models

from src import settings


class _netVGGFeatures(nn.Module):
    def __init__(self):
        super(_netVGGFeatures, self).__init__()
        self.vggnet = models.vgg16(pretrained=True).to(settings.device)
        self.vggnet.eval()
        self.layer_ids = [2, 7, 12, 21, 30]

    def main(self, z, levels):
        with torch.no_grad():
            layer_ids = self.layer_ids[:levels]
            id_max = layer_ids[-1] + 1
            output = []
            for i in range(id_max):
                z = self.vggnet.features[i](z)
                if i in layer_ids:
                    output.append(z)
        return output

    def forward(self, z, levels):
        output = self.main(z, levels)
        return output


_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).to(settings.device)
_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).to(settings.device)


class VGGDistance(nn.Module):
    def __init__(self, alpha, beta, powalpha, powbeta, levels=4):
        super(VGGDistance, self).__init__()
        self.vgg = _netVGGFeatures()
        self.levels = levels
        self.alpha = alpha
        self.beta = beta
        self.powalpha = powalpha
        self.powbeta = powbeta

    def norm(self, x):
        x_ = x.clone()

        for i in range(3):
            x_[:, i, :, :].sub_(_mean[i]).div_(_std[i])

        return x_

    def forward(self, x, y):
        l1loss = torch.abs(x - y).mean().unsqueeze_(0)

        x = self.norm(x)
        y = self.norm(y)
        f1 = self.vgg(x, self.levels)
        f2 = self.vgg(y, self.levels)
        featuresloss = torch.zeros(1, dtype=torch.float32).to(settings.device)
        for i in range(self.levels):
            layer_loss = torch.abs(f1[i] - f2[i]).mean()
            featuresloss.add_(layer_loss)

        l1loss.pow_(self.powalpha).mul_(self.alpha).add_(featuresloss.pow_(self.powbeta).mul_(self.beta))
        return l1loss
