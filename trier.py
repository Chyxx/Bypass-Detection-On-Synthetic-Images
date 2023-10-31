import torch.nn as nn


class Trier(nn.Module):
    def __init__(self, args):
        super(Trier, self).__init__()
        if args.trier_model == "parallel_noise":
            from models.trier.parallel_noise import ParallelNoise
            self.model = ParallelNoise(args)
        elif args.trier_model == "try_by_svd":
            from models.trier.try_by_svd import TryBySVD
            self.model = TryBySVD(args)
        elif args.trier_model == "try_by_gs":
            from models.trier.try_by_gs import TryByGS
            self.model = TryByGS(args)

    def forward(self, imgs, noise, d_net):
        return self.model(imgs, noise, d_net)
