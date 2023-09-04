import argparse
from importlib import import_module

import torch.nn as nn
import torch

def str2bool(v: str, strict=True) -> bool:
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ("true", "yes", "on" "t", "y", "1"):
            return True
        elif v.lower() in ("false", "no", "off", "f", "n", "0"):
            return False
    if strict:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")
    else:
        return True


def get_network(arch: str, isTrain=False, continue_train=False, init_gain=0.02, pretrained=True):
    if "resnet" in arch:
        from networks.resnet import ResNet

        resnet = getattr(import_module("networks.resnet"), arch)
        if isTrain:
            if continue_train:
                model: ResNet = resnet(num_classes=1)
            else:
                model: ResNet = resnet(pretrained=pretrained)
                model.fc = nn.Linear(2048, 1)
                nn.init.normal_(model.fc.weight.data, 0.0, init_gain)
        else:
            model: ResNet = resnet(num_classes=1)
        return model
    else:
        raise ValueError(f"Unsupported arch: {arch}")

