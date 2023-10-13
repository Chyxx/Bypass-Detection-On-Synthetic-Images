from config import opt

import time

import torch.optim as optim
from torch.utils.data import DataLoader
from visdom import Visdom

import torch.nn.functional as F
import numpy as np
import torch
import pytorch_msssim as PM

from processor import Processor
from trier import Trier
from detector import Detector
import dataset

seed = 1
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def norm(x):
    # Limit value between 0 and 1
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x = torch.where(x > 1, one, x)
    x = torch.where(x < 0, zero, x)
    return x


def try_noise(noise, r):
    # 采用高斯分布（旧版本采用泊松分布，往往使训练趋向极端化）
    t = torch.randn_like(noise)
    t = (t - t.mean()) / torch.var(t) * r
    return t


def main():
    data = dataset.ProcessorDataset(opt.file, opt.img_size)
    train_data, val_data = torch.utils.data.random_split(data, [len(data) - 8000, 8000])
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False)

    p_net = Processor(opt).cuda()
    d_net = Detector(opt).cuda()
    d_net.eval()
    t_net = Trier(opt).cuda()
    optimizer = optim.AdamW(p_net.parameters(), lr=opt.lr)
    wind = Visdom()
    wind.line([0.], [0], win="prob2/prob1", opts=dict(title="prob2/prob1"))
    wind.line([0.], [0], win="loss", opts=dict(title="loss"))

    for epoch in range(opt.n_epochs):
        t = time.perf_counter()
        for i, imgs in enumerate(train_loader):
            # train
            p_net.train()
            imgs = imgs.cuda()
            noise = p_net(imgs)
            p_imgs = norm(imgs + noise)
            with torch.no_grad():
                weighted_direction = t_net(imgs, noise, d_net)
                # 噪声的优化目标 = 当前噪声 + 梯度的反方向 * 步长
                target_noise = noise + weighted_direction * opt.delta
                # 优化目标的prob，理论上低于原来的prob（随着训练的进行，这个值往往会高于原prob）
                prob1 = d_net(imgs).cpu()
                prob2 = d_net(p_imgs).cpu()
                prob3 = d_net(norm(target_noise + imgs)).cpu()

            ssim = PM.ssim(p_imgs, imgs)
            loss = (
                    1e3 * (weighted_direction * (target_noise - noise)).mean()
                # + F.l1_loss(p_imgs, imgs) * opt.alpha
                # + F.mse_loss(p_imgs, imgs) * opt.beta
                # + torch.norm(noise, p=np.inf)
                # + (1.0 / ssim - 1) * opt.gamma
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                tt = time.perf_counter()
                # print
                print(
                    "epoch: {}, i: {}\nprob1: {}, prob2: {}, prob3: {}\nloss: {}, ssim: {}, t: {}s".format(
                        epoch, i, prob1.mean(), prob2.mean(), prob3.mean(), loss.item(), ssim,
                        tt - t
                    ))
                wind.line([loss.item()], [epoch * len(train_loader) + i],
                          win="loss", opts=dict(title="loss"), update="append")
                t = tt

            if i % (len(train_loader) // 10) == 0 and (epoch != 0 or i != 0):
                p_net.save(optimizer, epoch, i)

                # test
                t_val = time.perf_counter()
                p_net.eval()
                with torch.no_grad():
                    avg_prob1 = 0
                    avg_prob2 = 0
                    avg_ssim = 0
                    for j, imgs in enumerate(val_loader):
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
                    e = epoch + (i / (len(train_loader)//10)) * 0.1
                    wind.line([(avg_prob2 / avg_prob1).cpu()], [e],
                              win="prob2/prob1", opts=dict(title="prob2/prob1"), update="append")

                    # image
                    wind.images(imgs.cpu(), win="imgs")
                    wind.images(p_imgs.cpu(), win="p_imgs")
                    # print
                    print("epoch: {}, i: {}\navg_prob1: {}, avg_prob2: {}, avg_prob2/prob1: {}"
                          "\navg_ssim: {}".format(
                        epoch, i, avg_prob1, avg_prob2, avg_prob2 / avg_prob1, avg_ssim))

                print(str(time.perf_counter() - t_val) + "s")


if __name__ == "__main__":
    main()
