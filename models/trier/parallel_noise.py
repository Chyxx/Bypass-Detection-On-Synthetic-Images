import torch
import torch.nn as nn
from numpy import sqrt
import sys
sys.path.append("../..")
from utils.utils import norm, hinge_func


class ParallelNoise(nn.Module):
    def __init__(self, args, d_net):
        super(ParallelNoise, self).__init__()
        self.args = args
        self.d_net = d_net

    def func(self, x):
        return hinge_func(self.d_net(x), self.args.class_count, 0)

    def try_noise(self, noise):
        # 采用高斯分布（旧版本采用泊松分布，往往使训练趋向极端化）
        # return self.model(noise + self.model(torch.randn_like(noise)) * self.args.sigma) * self.args.epsilon - noise
        tmp = torch.rand_like(noise) - 0.5
        tmp /= torch.norm(tmp, dim=[2, 3], keepdim=True)
        tmp *= sqrt(tmp.size(2)*tmp.size(3)) * self.args.sigma
        return tmp

    def forward(self, imgs, noise):
        with torch.no_grad():
            p_imgs = norm(imgs + noise)
            # Test the detector with extra noise to estimate the gradient direction
            fx = self.func(p_imgs).squeeze()
            # 以归一化的prob差值为权重，对试探噪声的加权平均
            weighted_direction = torch.zeros([imgs.size(0), self.args.image_channels, self.args.img_size, self.args.img_size], device="cuda")
            # 规范化系数
            norm_c = torch.zeros(imgs.size(0), device="cuda")
            for j in range(self.args.num):
                # 一次试探
                t_noise = self.try_noise(noise)
                added_prob = self.func(norm(imgs + noise + t_noise)).squeeze()  # 求试探后的prob
                # 计算试探后的加权方向向量
                c = fx - added_prob
                t_noise *= c.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
                norm_c += c ** 2
                weighted_direction += t_noise
            # 对加权方向向量的prob值归一化，保证loss值稳定性
            norm_c = torch.sqrt(norm_c) + 1e-20
            weighted_direction /= norm_c.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
            # 对加权方向向量的距离（高斯分布的标准差）归一化，得到梯度近似值的负值
            weighted_direction /= self.args.sigma
            return weighted_direction
