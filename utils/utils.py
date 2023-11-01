import torch


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def norm(x):
    """Limit value between 0 and 1"""
    return torch.clamp(x, 0, 1)


def hinge_func(x, class_count, t):
    """目标攻击时的损失函数"""
    k = 0
    # label = one_hot(torch.full([x.size(0)], t, dtype=torch.int), class_count).type(torch.int)
    # tmp = torch.where(label == 1, float('-inf'), x)
    # tmp = torch.log(tmp.max(dim=1)[0]) - torch.log(x[:, t])
    # return torch.where(tmp > -k, tmp, -k)
    idx = list(range(class_count))
    idx.pop(t)
    idx = torch.tensor(idx).cuda()
    tmp = torch.log(x.index_select(1, idx).max(dim=1)[0]) - torch.log(x[:, t])
    return torch.clamp(tmp, min=-k)


def get_prob(x):
    return torch.chunk(x, dim=1, chunks=2)[1]
