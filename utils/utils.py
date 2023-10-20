import torch


def one_hot(x, class_count):
    x = x.cpu()
    return torch.eye(class_count)[x, :].cuda()