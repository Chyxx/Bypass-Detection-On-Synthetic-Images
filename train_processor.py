import argparse
import os.path
import numpy as np

import torchvision.datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from visdom import Visdom


import torch

from utils.utils import str2bool

from processor import Processor
from detector import Detector


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--epsilon", type=float, default=5e-3, help="coefficient in the loss function")
parser.add_argument(
    "-f", "--file", default="E:\data", type=str, help="path to data directory"
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="data/detector-ckpt/imagenet_adm.pth",
)
parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument("--aug_norm", type=str2bool, default=False)
parser.add_argument("--image_channels", type=int, default=3, help="number of image channels")
parser.add_argument("--model_channels", type=int, default=64, help="number of model channels")
opt = parser.parse_args()


transform = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
test_data = torchvision.datasets.ImageFolder(root=os.path.join(opt.file, "sdv2-test"), transform=transform)
test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True)
train_data = torchvision.datasets.ImageFolder(root=os.path.join(opt.file, "sdv2-train"), transform=transform)
train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)


def norm(x):
    # Limit the data range to between 0 and 1
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x = torch.where(x > 1, one, x)
    x = torch.where(x < 0, zero, x)
    return x


def main():
    p_net = Processor(opt).cuda()
    d_net = Detector(opt).cuda()
    d_net.eval()
    optimizer = optim.Adam(p_net.parameters(), lr=opt.lr)

    wind = Visdom()
    wind.line([0.], [0], win="prob2/prob1", opts=dict(title="prob2/prob1"))
    wind.line([0.], [0], win="loss", opts=dict(title="loss"))

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(train_loader):
            # train
            p_net.train()
            imgs = imgs.cuda()
            with torch.no_grad():
                prob1 = d_net(imgs)
            noise = p_net(imgs)
            p_imgs = norm(imgs + noise)
            prob2 = d_net(p_imgs)
            loss = (prob2 / prob1).mean()
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch: {}, i: {}\nprob1: {}, prob2: {}\nloss: {}".format(
                epoch, i, prob1.mean(), prob2.mean(), loss.item()))

            if i % 10 == 0:
                # test
                with torch.no_grad():
                    p_net.eval()
                    noise = p_net(imgs)
                    p_imgs = norm(imgs + noise)
                    prob2 = d_net(p_imgs)
                    loss = (prob2 / prob1).mean()

                wind.images(imgs.cpu(), win="imgs")
                wind.images(p_imgs.cpu(), win="p_imgs")
                wind.line([prob2.mean().detach().cpu()/prob1.mean().detach().cpu()], [epoch * len(train_loader) + i],
                          win="prob2/prob1", opts=dict(title="prob2/prob1"), update="append")
                wind.line([loss.item()], [epoch * len(train_loader) + i],
                          win="loss", opts=dict(title="loss"), update="append")

            if i % 200 == 0:
                # save
                checkpoint = {"model_state_dict": p_net.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                path_checkpoint = "data/processor-ckpt/{}_{}_.pkl".format(epoch, i)
                torch.save(checkpoint, path_checkpoint)



if __name__ == "__main__":
    main()



