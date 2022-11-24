import torch


def get_optimizer(model, lr, momentum=0.9, weight_decay=1e-4):
    return torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_lr_scheduler(optimizer, step_size):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1, last_epoch=-1)