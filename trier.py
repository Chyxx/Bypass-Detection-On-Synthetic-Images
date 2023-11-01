import torch.nn as nn


class Trier(nn.Module):
    def __init__(self, args, model_type, d_net):
        super(Trier, self).__init__()
        if model_type == "parallel_noise":
            from models.trier.parallel_noise import ParallelNoise
            self.model = ParallelNoise(args, d_net)
        elif model_type == "try_by_svd":
            from models.trier.try_by_svd import TryBySVD
            self.model = TryBySVD(args, d_net)
        elif model_type == "try_by_gs":
            from models.trier.try_by_gs import TryByGS
            self.model = TryByGS(args, d_net)

    def forward(self, imgs, noise):
        return self.model(imgs, noise)
