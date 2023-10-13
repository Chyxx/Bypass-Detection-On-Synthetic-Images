import torch
import torch.nn as nn


def try_noise(noise, r):
    # 采用高斯分布（旧版本采用泊松分布，往往使训练趋向极端化）
    t = torch.randn_like(noise)
    t = (t - t.mean(dim=1, keepdim=True)) / torch.sqrt(torch.var(t, dim=1, keepdim=True)) * r
    return t


def norm(x):
    # Limit value between 0 and 1
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x = torch.where(x > 1, one, x)
    x = torch.where(x < 0, zero, x)
    return x


class ParallelNoise(nn.Module):
    def __init__(self, args):
        super(ParallelNoise, self).__init__()
        self.args = args

    def forward(self, imgs, noise, d_net):
        with torch.no_grad():
            p_imgs = norm(imgs + noise)
            # Test the detector with extra noise to estimate the gradient direction
            prob2 = d_net(p_imgs).cpu()
            r = self.args.sigma
            avg_prob3 = torch.zeros(imgs.size(0))
            # 以prob差值为权重，对试探噪声的加权平均
            weighted_direction = torch.zeros([imgs.size(0), self.args.image_channels, self.args.img_size, self.args.img_size]).cuda()
            # 规范化系数
            norm_c = torch.zeros(imgs.size(0))
            for j in range(self.args.num):
                # 一次试探
                t_noise = try_noise(noise, r)
                added_imgs = norm(imgs + noise + t_noise)
                added_prob = d_net(added_imgs).squeeze().cpu().numpy()  # 求试探后的prob
                # 计算试探后的平均porb，平均方向向量，加权方向向量
                avg_prob3 += added_prob / self.args.num
                for k in range(imgs.size(0)):
                    t_noise[k] *= (prob2[k] - added_prob[k]).item()
                    norm_c[k] += (prob2[k] - added_prob[k]).item() ** 2
                weighted_direction += t_noise / self.args.num
            # 对加权方向向量的prob值归一化，保证loss值稳定性
            norm_c = torch.sqrt(norm_c) + 1e-10
            for j in range(imgs.size(0)):
                weighted_direction[j] /= norm_c[j].item()
            # 利用软阈值函数对加权方向向量进行收缩处理，排除干扰噪声，保留目标方向。
            # abs = torch.abs(weighted_direction)
            # tau = abs.mean(dim=(2, 3)).unsqueeze(dim=2).unsqueeze(dim=3) * 0.5
            # sub = abs - tau
            # zeros = sub - sub
            # n_sub = torch.max(sub, zeros)
            # weighted_direction = torch.mul(torch.sign(weighted_direction), n_sub)
            # 对加权方向向量的距离（高斯分布的方差）归一化，得到梯度近似值的负值
            weighted_direction /= self.args.sigma
            return weighted_direction
