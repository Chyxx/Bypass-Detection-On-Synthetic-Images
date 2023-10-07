import torch

import torch.nn as nn
from torch.nn import functional as F

from torchvision.models import resnet50, ResNet50_Weights


class Detector(nn.Module):
    def __init__(self, args):
        super(Detector, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_ftrs, 2))
        self.model = self.model.cuda()
        checkpoint = torch.load(args.detector_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        return torch.chunk(F.softmax(self.model(x), dim=1), dim=1, chunks=2)[1]
