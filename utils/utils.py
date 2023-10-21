import torch


def one_hot(x, class_count):
    x = x.cpu()
    return torch.eye(class_count)[x, :].cuda()


def norm(x):
    """Limit value between 0 and 1"""
    return torch.clamp(x, 0, 1)