import os

import torch.nn as nn
import torch


class Processor(nn.Module):
    def __init__(self, args):
        super(Processor, self).__init__()
        if args.processor_model == "dncnn":
            from models.processor.dncnn import DnCNN
            self.model = DnCNN(args)

        if args.processor_path != "":
            checkpoint = torch.load(args.detector_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.path = args.processor_save_path

    def forward(self, x):
        return self.model(x)

    def save(self, optimizer, epoch, i):
        checkpoint = {"model_state_dict": self.model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "i": i}
        torch.save(checkpoint, os.path.join(self.path, "_{}_{}_.pkl".format(epoch, i)))
