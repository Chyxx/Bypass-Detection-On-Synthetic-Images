import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DetectorDataset(Dataset):
    def __init__(self, root, resize):
        super(DetectorDataset, self).__init__()
        self.root = root
        self.resize = resize
        self.imgs = []
        self.labels = []

        # REAL
        imgs = glob.glob(os.path.join(self.root + "/nature", "*.JPEG"))
        for i in range(len(imgs)):
            self.imgs.append(imgs[i])
            self.labels.append(0)

        # FAKE
        imgs = glob.glob(os.path.join(self.root + "/ai", "*.png"))
        for i in range(len(imgs)):
            self.imgs.append(imgs[i])
            self.labels.append(1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, label = self.imgs[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize), int(self.resize))),
            # transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            # transforms.RandomRotation(15),
            # transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
        ])

        img = tf(img)
        label = torch.tensor(label)
        return img, label


class ProcessorDataset(Dataset):
    def __init__(self, root, resize):
        super(ProcessorDataset, self).__init__()
        self.root = root
        self.imgs = []
        self.resize = resize

        # FAKE
        imgs = glob.glob(os.path.join(self.root + "/ai", "*.png"))
        for i in range(len(imgs)):
            self.imgs.append(imgs[i])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            # transforms.RandomCrop((int(self.resize), int(self.resize))),
            transforms.Resize((int(self.resize), int(self.resize))),
            transforms.ToTensor(),
        ])

        img = tf(img)
        return img
