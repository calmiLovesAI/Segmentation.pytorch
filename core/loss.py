import torch


def cross_entropy(input, target):
    loss = torch.nn.functional.cross_entropy(input, target, reduction="none").mean(dim=1).mean(dim=1)
    return torch.mean(loss)