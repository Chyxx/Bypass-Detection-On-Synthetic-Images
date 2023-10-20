import torch
import torch.nn as nn


def try_noise(noise, sigma):
    # 采用高斯分布（旧版本采用泊松分布，往往使训练趋向极端化）
    t = torch.randn_like(noise)
    t = sigma * (t - t.mean(dim=(1, 2, 3), keepdim=True)) / torch.sqrt(torch.var(t, dim=(1, 2, 3), keepdim=True))
    return t

class ParallelNoise(nn.Module):
    def __init__(self, args):
        super(ParallelNoise, self).__init__()
        self.args = args

    def forward(self, imgs, noise, d_net):
        with torch.no_grad():
            p_imgs = torch.clamp(imgs + noise,0.0,1.0)
            # Test the detector with extra noise to estimate the gradient direction
            prob2 = d_net(p_imgs)
            sigma = self.args.sigma
            # 以归一化的prob差值为权重，对试探噪声的加权平均
            weighted_direction = torch.zeros([imgs.size(0), self.args.image_channels, self.args.img_size, self.args.img_size]).cuda()
            # 规范化系数
            norm_c = torch.zeros(imgs.size(0))
            for j in range(self.args.num):
                # 一次试探
                t_noise = try_noise(noise, sigma)
                added_imgs = torch.clamp(imgs + noise + t_noise,0.0,1.0)
                added_prob = d_net(added_imgs).squeeze().cpu().numpy()  # 求试探后的prob
                # 计算试探后的加权方向向量
                for k in range(imgs.size(0)):
                    t_noise[k] *= (prob2[k] - added_prob[k]).item()
                    norm_c[k] += (prob2[k] - added_prob[k]).item() ** 2
                weighted_direction += t_noise
            # 对加权方向向量的prob值归一化，保证loss值稳定性
            norm_c = torch.sqrt(norm_c) + 1e-20
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
            weighted_direction /= sigma
            return weighted_direction
