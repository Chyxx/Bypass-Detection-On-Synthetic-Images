import os

import torch

import torch.nn as nn
from torch.nn import functional as F


class Detector(nn.Module):
    def __init__(self, args):
        super(Detector, self).__init__()
        if args.detector_model == "resnet50":
            from models.detector.resnet50 import ResNet50
            self.model = ResNet50(args)

        if args.detector_path != "":
            checkpoint = torch.load(args.detector_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        return torch.chunk(F.softmax(self.model(x), dim=1), dim=1, chunks=2)[1]


class DetectorToTrain(nn.Module):
    def __init__(self, args):
        super(DetectorToTrain, self).__init__()
        if args.detector_model == "resnet50":
            from models.detector.resnet50 import ResNet50
            self.model = ResNet50(args)

        if args.detector_path != "":
            checkpoint = torch.load(args.detector_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.path = args.detector_save_path

    def forward(self, x):
        return self.model(x)

    def save(self, optimizer, epoch, i):
        checkpoint = {"model_state_dict": self.model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "i": i}
        torch.save(checkpoint, os.path.join(self.path, "_{}_{}_.pkl".format(epoch, i)))