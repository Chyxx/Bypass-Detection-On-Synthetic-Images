import torch


def one_hot(x, class_count):
    x = x.cpu()
    return torch.eye(class_count)[x, :].cuda()


def norm(x):
    # Limit value between 0 and 1
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x = torch.where(x > 1, one, x)
    x = torch.where(x < 0, zero, x)
    return x