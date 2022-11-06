import torch
import argparse

from core.fcn_train import train_loop, evaluate_loop
from core.parse import update_cfg
from utils.tools import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='yaml path')
    parser.add_argument('--mode', type=str, default='train', help='train mode or valid mode')
    parser.add_argument('--ckpt', type=str, default='', help='model checkpoint path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(filepath=[args.cfg])
    cfg = update_cfg(cfg, device)

    model = cfg["model"]
    model.to(device=device)

    train_dataloader = cfg["train_dataloader"]
    valid_dataloader = cfg["valid_dataloader"]

    if args.mode == "train":
        train_loop(cfg, model, train_dataloader, valid_dataloader)
    if args.mode == "valid":
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        evaluate_loop(cfg, model, valid_dataloader)


if __name__ == '__main__':
    main()
