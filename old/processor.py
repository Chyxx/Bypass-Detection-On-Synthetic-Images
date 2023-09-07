import torch

import torch.nn as nn
import torch.nn.functional as f


class Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out, momentum=0.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Processor(nn.Module):
    """
    Similar to DnCNN, which is denoising algorithm.But we're adding noise here.
    """
    def __init__(self, args):
        super(Processor, self).__init__()
        self.args = args
        self.model_1 = nn.Sequential(
            nn.Conv2d(in_channels=args.image_channels, out_channels=args.model_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Block(args.model_channels, args.model_channels),
            Block(args.model_channels, args.model_channels),
            Block(args.model_channels, args.model_channels),
            Block(args.model_channels, args.model_channels),
            nn.Conv2d(in_channels=args.model_channels, out_channels=args.image_channels*2,
                      kernel_size=3, stride=1, padding=1),
        )
        self.model_2 = nn.Sequential(
            nn.BatchNorm2d(args.image_channels, affine=False, track_running_stats=False)
        )

    def forward(self, x):
        ck = torch.chunk(self.model_1(x), 2, dim=1)
        y = ck[0] + ck[1] * torch.randn_like(ck[1])
        y = self.model_2(y) * self.args.epsilon  # The mean of the noise is zero and the variance is epsilon
        return y
