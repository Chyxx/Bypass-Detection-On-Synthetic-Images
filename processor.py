import torch.nn as nn
import torch

import models.processor


class Processor(nn.Module):
    def __init__(self, args):
        super(Processor, self).__init__()
        if args.processor_model == "dncnn":
            self.model = models.processor.dncnn.DnCNN(args)

        if args.processor_path != "":
            checkpoint = torch.load(args.detector_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        return self.model(x)
