import torch

import torch.nn as nn
from torch.nn import functional as F

from torchvision.models import resnet50, ResNet50_Weights


class ResNet50(nn.Module):
    def __init__(self, args):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 2))

    def forward(self, x):
        return self.model(x)
