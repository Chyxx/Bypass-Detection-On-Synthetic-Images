from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob


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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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
