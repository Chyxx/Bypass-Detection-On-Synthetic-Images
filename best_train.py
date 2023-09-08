import argparse
import os.path
import numpy as np
import time

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

from best_processor import Processor
from detector import Detector


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum of SGD")
parser.add_argument("--alpha", type=float, default=1e-5, help="coefficient of l1-loss")
parser.add_argument("--beta", type=float, default=1, help="coefficient of l2-loss")
parser.add_argument("--gamma", type=float, default=3e-3, help="coefficient of ssim-loss")
parser.add_argument("--epsilon", type=float, default=5e-3, help="variance of generated noise")
parser.add_argument("--sigma", type=float, default=1e-3, help="variance of extra noise")
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
parser.add_argument("--path", type=str, default="data/processor-ckpt/train_noise_1_0_.pkl")

opt = parser.parse_args()


transform = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

seed = 23
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

train_data = torchvision.datasets.ImageFolder(root=os.path.join(opt.file, "sdv2-train"), transform=transform)
train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
val_and_test_data = torchvision.datasets.ImageFolder(root=os.path.join(opt.file, "sdv2-val_and_test"), transform=transform)
val_data, test_data = torch.utils.data.random_split(val_and_test_data, [5000, 5000])
val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)


def norm(x):
    # Limit value between 0 and 1
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x = torch.where(x > 1, one, x)
    x = torch.where(x < 0, zero, x)
    return x


def try_noise(noise, r):
    t = torch.randn_like(torch.sigmoid(noise))
    t = (t - t.mean()) / torch.var(t) * r
    return t


def main():
    p_net = Processor(opt).cuda()
    # checkpoint = torch.load(opt.path)
    # p_net.load_state_dict(checkpoint['model_state_dict'])
    d_net = Detector(opt).cuda()
    d_net.eval()
    # optimizer = optim.SGD(p_net.parameters(), lr=opt.lr, momentum=opt.momentum)
    optimizer = optim.AdamW(p_net.parameters(), lr=opt.lr)
    wind = Visdom()
    wind.line([0.], [0], win="prob2/prob1", opts=dict(title="prob2/prob1"))
    wind.line([0.], [0], win="loss", opts=dict(title="loss"))

    for epoch in range(opt.n_epochs):
        t = time.perf_counter()
        for i, (imgs, _) in enumerate(train_loader):
            # train

            p_net.train()
            imgs = imgs.cuda()
            noise = p_net(imgs)
            p_imgs = norm(imgs + noise)

            with torch.no_grad():
                # Test the detector with extra noise to estimate the gradient direction
                prob1 = d_net(imgs).cpu()
                prob2 = d_net(p_imgs).cpu()
                r = opt.sigma
                avg_prob3 = torch.zeros(imgs.size(0))
                avg_direction = torch.zeros([imgs.size(0), opt.image_channels, opt.img_size, opt.img_size]).cuda()
                weighted_direction = torch.zeros([imgs.size(0), opt.image_channels, opt.img_size, opt.img_size]).cuda()
                for j in range(opt.num):
                    t_noise = try_noise(noise, r)
                    added_imgs = norm(imgs + noise + t_noise)
                    added_prob = d_net(added_imgs).squeeze().cpu().numpy()
                    avg_prob3 += added_prob / opt.num
                    avg_direction += t_noise
                    for k in range(imgs.size(0)):
                        t_noise[k] *= (prob2[k] - added_prob[k]).item()
                    weighted_direction += t_noise
                for j in range(imgs.size(0)):
                    avg_direction[j] *= (prob2[j] - avg_prob3[j]).item()
                weighted_direction -= avg_direction
                target_noise = noise + weighted_direction * 3000./opt.num
                best_prob3 = d_net(norm(target_noise + imgs))

            ssim = PM.ssim(p_imgs, imgs)
            loss = (
                    10000*(weighted_direction * (target_noise - noise)).mean()
                    # F.mse_loss(noise, weighted_noise)
                    + F.l1_loss(p_imgs, imgs) * opt.alpha
                    # + F.mse_loss(p_imgs, imgs) * opt.beta
                    + (1.0 / ssim - 1) * opt.gamma
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:

                tt = time.perf_counter()
                # print
                print("epoch: {}, i: {}\nprob1: {}, prob2: {}, best_prob3: {}, avg_prob3: {}\nloss: {}, ssim: {}, t: {}s".format(
                    epoch, i, prob1.mean(), prob2.mean(), best_prob3.mean(), avg_prob3.mean(), loss.item(), ssim, tt - t
                ))
                wind.line([loss.item()], [epoch * len(train_loader) + i],
                          win="loss", opts=dict(title="loss"), update="append")
                t = tt

            if i % 1000 == 0 and (epoch != 0 or i != 0):
                checkpoint = {"model_state_dict": p_net.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                path_checkpoint = "data/processor-ckpt/train_noise_{}_{}_.pkl".format(epoch, i)
                torch.save(checkpoint, path_checkpoint)

                # test
                t_val = time.perf_counter()
                p_net.eval()
                with torch.no_grad():
                    avg_prob1 = 0
                    avg_prob2 = 0
                    avg_ssim = 0
                    for j, (imgs, _) in enumerate(val_loader):
                        imgs = imgs.cuda()
                        noise = p_net(imgs)
                        p_imgs = norm(imgs + noise)
                        prob1 = d_net(imgs)
                        prob2 = d_net(p_imgs)
                        ssim = PM.ssim(p_imgs, imgs)
                        avg_prob1 += prob1.mean() * imgs.size(0) / len(val_data)
                        avg_prob2 += prob2.mean() * imgs.size(0) / len(val_data)
                        avg_ssim += ssim / len(val_loader)

                    # line
                    e = epoch if i == 0 else epoch + 0.5
                    wind.line([(avg_prob2/avg_prob1).cpu()], [e],
                              win="prob2/prob1", opts=dict(title="prob2/prob1"), update="append")

                    # image
                    wind.images(imgs.cpu(), win="imgs")
                    wind.images(p_imgs.cpu(), win="p_imgs")
                    wind.images(norm(noise), win="noise")
                    # print
                    print("epoch: {}, i: {}\navg_prob1: {}, avg_prob2: {}, avg_prob2/prob1: {}"
                          "\navg_ssim: {}".format(
                        epoch, i, avg_prob1, avg_prob2, avg_prob2/avg_prob1, avg_ssim))

                print(str(time.perf_counter() - t_val) + "s")


if __name__ == "__main__":
    main()


