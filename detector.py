import torch

import torch.nn as nn
from torch.nn import functional as F
import models.detector


class Detector(nn.Module):
    def __init__(self, args):
        super(Detector, self).__init__()
        if args.detector_model == "resnet50":
            self.model = models.detector.resnet50.ResNet50

        if args.detector_path != "":
            checkpoint = torch.load(args.detector_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        return torch.chunk(F.softmax(self.model(x), dim=1), dim=1, chunks=2)[1]


class DetectorToTrain(nn.Module):
    def __init__(self, args):
        super(DetectorToTrain, self).__init__()
        if args.detector_model == "resnet50":
            self.model = models.detector.resnet50.ResNet50

        if args.detector_path != "":
            checkpoint = torch.load(args.detector_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        return self.model(x)

