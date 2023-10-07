import glob
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from utils.utils import get_network

class Detector(nn.Module):
    def __init__(self, args):
        super(Detector, self).__init__()
        self.model = get_network(args.arch)
        self.args = args
        state_dict = torch.load(args.model_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)

    def forward(self, imgs):
        if self.args.aug_norm:
            imgs = TF.normalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if len(imgs.size()) == 4:
            prob = self.model(imgs).sigmoid()
        else:
            prob = self.model(imgs.unsqueeze(0)).squeeze(0).sigmoid()
        return prob
