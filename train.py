import torch
import argparse

from torch import nn

from core.base_train import train_loop, evaluate_loop
from core.loss import FocalLoss
from core.parse import update_cfg
from utils.tools import load_yaml, Saver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='yaml path')
    parser.add_argument('--mode', type=str, default='train', help='train mode or valid mode')
    parser.add_argument('--ckpt', type=str, default='', help='model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0)
    parser.add_argument('--step_size', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='', choices=["ce", 'focal'])
    opts = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    cfg = load_yaml(filepath=[opts.cfg])
    cfg = update_cfg(cfg, opts, device)

    model = cfg["model"]
    model.to(device=device)

    train_dataloader = cfg["train_dataloader"]
    valid_dataloader = cfg["valid_dataloader"]

    loss_type = cfg["Train"]["loss_type"]
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(reduction='mean')
    elif loss_type == "focal":
        criterion = FocalLoss()

    if opts.mode == "train":
        train_loop(cfg, model, criterion, train_dataloader, valid_dataloader)
    if opts.mode == "valid":
        Saver.load_ckpt(model, opts.ckpt, device)
        evaluate_loop(cfg, model, criterion, valid_dataloader)


if __name__ == '__main__':
    main()
