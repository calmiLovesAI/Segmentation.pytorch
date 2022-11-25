from typing import List

import torch


def get_optimizer(model, lr, momentum=0.9, weight_decay=1e-4):
    return torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_lr_scheduler(optimizer, milestones: List[int], gamma: int):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)