import argparse
import os.path
import numpy as np

import torchvision.datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from visdom import Visdom


import torch
import pytorch_msssim as PM

from utils.utils import str2bool

from multiprocessor import Processor
from detector import Detector


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--alpha", type=float, default=1e-6, help="coefficient of l1-loss")
parser.add_argument("--beta", type=float, default=1, help="coefficient of l2-loss")
parser.add_argument("--gamma", type=float, default=1e-2, help="coefficient of ssim-loss")
parser.add_argument("--epsilon", type=float, default=5e-2, help="variance of generated noise")
parser.add_argument("--sigma", type=float, default=2e-2, help="variance of extra noise")
parser.add_argument("--num", type=int, default=20, help="number of try noise")
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


def take_second(elem):
    return elem[1]


def norm(x):
    # Limit value between 0 and 1
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x = torch.where(x > 1, one, x)
    x = torch.where(x < 0, zero, x)
    return x


def try_noise(noise, r):
    t = torch.poisson(F.sigmoid(noise))
    t = (t - t.mean()) / torch.var(t) * r
    return t


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
            noise = p_net(imgs)
            p_imgs = norm(imgs + noise)

            with torch.no_grad():
                # Test the detector with extra noise to estimate the gradient direction
                prob1 = d_net(imgs)
                prob2 = d_net(p_imgs)
                r = opt.sigma
                weighted_direction = torch.zeros([imgs.size(0), opt.image_channels, opt.img_size, opt.img_size]).cuda()
                for j in range(opt.num):
                    t_noise = try_noise(noise, r)
                    added_imgs = norm(imgs + noise + t_noise)
                    added_prob = d_net(added_imgs).squeeze().cpu().numpy()
                    for k in range(imgs.size(0)):
                        t_noise[k] *= (prob2[k] - added_prob[k])
                    weighted_direction += t_noise
                target_noise = noise + weighted_direction * 3./opt.num
                prob3 = d_net(norm(target_noise + imgs))

            ssim = PM.ssim(p_imgs, imgs)
            loss = (
                    (weighted_direction * (target_noise - noise)).mean()
                    # F.mse_loss(noise, weighted_noise)
                    # + F.l1_loss(p_imgs, imgs) * opt.alpha
                    # + F.mse_loss(p_imgs, imgs) * opt.beta
                    + (1.0 / ssim - 1) * opt.gamma
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch: {}, i: {}\nprob1: {}, prob2: {}, prob3: {}\nloss: {}, ssim: {}".format(
                epoch, i, prob1.mean(), prob2.mean(), prob3.mean(), loss.item(), ssim
            ))
            if i % 5 == 0:
                wind.line([prob2.mean().detach().cpu()/prob1.mean().detach().cpu()], [epoch * len(train_loader) + i],
                          win="prob2/prob1", opts=dict(title="prob2/prob1"), update="append")
                wind.line([loss.item()], [epoch * len(train_loader) + i],
                          win="loss", opts=dict(title="loss"), update="append")

            if i % 10 == 0:
                wind.images(imgs.cpu(), win="imgs")
                wind.images(p_imgs.cpu(), win="p_imgs")

            if i % 200 == 0:
                checkpoint = {"model_state_dict": p_net.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                path_checkpoint = "data/processor-ckpt/best_noise_{}_{}_.pkl".format(epoch, i)
                torch.save(checkpoint, path_checkpoint)


if __name__ == "__main__":
    main()


