import torch
import torch.nn as nn


def norm(x):
    """Limit value between 0 and 1"""
    return torch.clamp(x, 0, 1)


class ParallelNoise(nn.Module):
    def __init__(self, args):
        super(ParallelNoise, self).__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.LayerNorm([args.img_size, args.img_size], elementwise_affine=False, eps=1e-40)
        )

    def try_noise(self, noise):
        # 采用高斯分布（旧版本采用泊松分布，往往使训练趋向极端化）
        return self.model(noise + self.model(torch.rand_like(noise)) * self.args.sigma) * self.args.epsilon - noise

    def forward(self, imgs, noise, d_net):
        with torch.no_grad():
            p_imgs = norm(imgs + noise)
            # Test the detector with extra noise to estimate the gradient direction
            prob2 = d_net(p_imgs).squeeze()
            # 以归一化的prob差值为权重，对试探噪声的加权平均
            weighted_direction = torch.zeros([imgs.size(0), self.args.image_channels, self.args.img_size, self.args.img_size], device="cuda")
            # 规范化系数
            norm_c = torch.zeros(imgs.size(0), device="cuda")
            for j in range(self.args.num):
                # 一次试探
                t_noise = self.try_noise(noise)
                added_prob = d_net(norm(imgs + noise + t_noise)).squeeze()  # 求试探后的prob
                # 计算试探后的加权方向向量
                c = prob2 - added_prob
                t_noise *= c.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
                norm_c += c ** 2
                weighted_direction += t_noise
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
