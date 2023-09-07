import torch.nn as nn


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
    和DnCNN类似的思路，但这里是给图像加上噪声
    """
    def __init__(self, args):
        super(Processor, self).__init__()
        self.args = args
        self.module_1 = nn.Sequential(
            nn.Conv2d(in_channels=args.image_channels, out_channels=args.model_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Block(args.model_channels, args.model_channels),
            Block(args.model_channels, args.model_channels),
            Block(args.model_channels, args.model_channels),
            Block(args.model_channels, args.model_channels),
            Block(args.model_channels, args.model_channels),
            nn.Conv2d(in_channels=args.model_channels, out_channels=args.image_channels,
                      kernel_size=3, stride=1, padding=1),
        )
        self.module_2 = nn.Sequential(
            nn.LayerNorm([args.img_size, args.img_size], elementwise_affine=False)
        )

    def forward(self, x):
        y = self.module_2(self.module_1(x)) * self.args.epsilon
        return y