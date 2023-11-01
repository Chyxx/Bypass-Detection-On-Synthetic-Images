import torch
import torch.nn as nn
from numpy import sqrt
import sys
sys.path.append("../..")
from utils.utils import norm, hinge_func

def norm(x):
    """Limit value between 0 and 1"""
    return torch.clamp(x, 0, 1)


class TryByGS(nn.Module):
    def __init__(self, args, d_net):
        super(TryByGS, self).__init__()
        self.args = args
        self.d_net = d_net

    def func(self, x):
        return hinge_func(self.d_net(x), self.args.class_count, 0)

    def gram_schmidt(self):
        rand_noise = torch.rand([self.args.num, self.args.image_channels, self.args.img_size, self.args.img_size]).cuda()
        for i in range(self.args.num):
            projection = torch.zeros_like(rand_noise[i]).cuda()
            for k in range(i):
                projection += (rand_noise[i]*rand_noise[k]).sum() * rand_noise[k]
            rand_noise[i] -= projection
            rand_noise[i] /= torch.sqrt((rand_noise[i]*rand_noise[i]).sum())
        rand_noise *= sqrt(self.args.image_channels * self.args.img_size**2) * self.args.sigma
        return rand_noise

    def forward(self, imgs, noise):
        with torch.no_grad():
            p_imgs = norm(imgs + noise)
            # Test the detector with extra noise to estimate the gradient direction
            fx = self.func(p_imgs).squeeze()
            # 以归一化的prob差值为权重，对试探噪声的加权平均
            weighted_direction = torch.zeros([imgs.size(0), self.args.image_channels, self.args.img_size, self.args.img_size]).cuda()
            # 规范化系数
            norm_c = torch.zeros(imgs.size(0)).cuda()
            gs_noise = self.gram_schmidt()
            for j in range(self.args.num):
                # 一次试探
                t_noise = gs_noise[j]
                added_prob = self.func(norm(imgs + noise + t_noise)).squeeze()  # 求试探后的prob
                # 计算试探后的加权方向向量
                c = fx - added_prob
                norm_c += c ** 2
                for k in range(imgs.size(0)):
                    weighted_direction[k] += t_noise * c[k]
            # 对加权方向向量的prob值归一化，保证loss值稳定性
            norm_c = torch.sqrt(norm_c) + 1e-20
            weighted_direction /= norm_c.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
            # 对加权方向向量的距离（高斯分布的标准差）归一化，得到梯度近似值的负值
            weighted_direction /= self.args.sigma
            return weighted_direction
