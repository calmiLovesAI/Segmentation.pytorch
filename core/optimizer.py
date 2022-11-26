from typing import List

import torch


def get_optimizer(model, optimizer_name: str, lr: float):
    match optimizer_name:
        case "SGD":
            return torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        case "AdamW":
            return torch.optim.AdamW(params=model.parameters(), lr=lr)
    raise NotImplementedError(f"Optimizer {optimizer_name} is not implemented")


def get_lr_scheduler(optimizer, scheduler_name: str, milestones: List[int], gamma: int):
    match scheduler_name:
        case "MultiStepLR":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1)
        case "None":
            return "None"
    return NotImplementedError(f"Lr scheduler {scheduler_name} is not implemented")
