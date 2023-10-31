import torch
import torch.nn as nn
from numpy import sqrt


def norm(x):
    # Limit value between 0 and 1
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x = torch.where(x > 1, one, x)
    x = torch.where(x < 0, zero, x)
    return x


class TryByGS(nn.Module):
    def __init__(self, args):
        super(TryByGS, self).__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.LayerNorm([args.img_size, args.img_size], elementwise_affine=False, eps=1e-40)
        )


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

    def forward(self, imgs, noise, d_net):
        with torch.no_grad():
            p_imgs = norm(imgs + noise)
            # Test the detector with extra noise to estimate the gradient direction
            prob2 = d_net(p_imgs).squeeze()
            # 以归一化的prob差值为权重，对试探噪声的加权平均
            weighted_direction = torch.zeros([imgs.size(0), self.args.image_channels, self.args.img_size, self.args.img_size]).cuda()
            # 规范化系数
            norm_c = torch.zeros(imgs.size(0)).cuda()
            gs_noise = self.gram_schmidt()
            for j in range(self.args.num):
                # 一次试探
                t_noise = gs_noise[j]
                added_prob = d_net(norm(imgs + noise + t_noise)).squeeze()  # 求试探后的prob
                # 计算试探后的加权方向向量
                c = prob2 - added_prob
                norm_c += c ** 2
                for k in range(imgs.size(0)):
                    weighted_direction[k] += t_noise * c[k]
            # 对加权方向向量的prob值归一化，保证loss值稳定性
            norm_c = torch.sqrt(norm_c) + 1e-20
            weighted_direction /= norm_c.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
            # 利用软阈值函数对加权方向向量进行收缩处理，排除干扰噪声，保留目标方向。
            # abs = torch.abs(weighted_direction)
            # tau = abs.mean(dim=(2, 3)).unsqueeze(dim=2).unsqueeze(dim=3) * 0.5
            # sub = abs - tau
            # zeros = sub - sub
            # n_sub = torch.max(sub, zeros)
            # weighted_direction = torch.mul(torch.sign(weighted_direction), n_sub)
            # 对加权方向向量的距离（高斯分布的标准差）归一化，得到梯度近似值的负值
            weighted_direction /= self.args.sigma
            return weighted_direction
