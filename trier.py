import torch
import torch.nn as nn

import models.trier


class Trier(nn.Module):
    def __init__(self, args):
        super(Trier, self).__init__()
        if args.trier_model == "parallel_noise":
            self.model = models.trier.parallel_noise.ParallelNoise(args)

    def forward(self, imgs, noise, d_net):
        return self.model(imgs, noise, d_net)
