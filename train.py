import torch
import argparse

from core.fcn_train import train_loop
from core.parse import update_cfg
from utils.tools import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='', help="yaml path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(filepath=[args.cfg])
    cfg = update_cfg(cfg, device)

    model = cfg["model"]
    model.to(device=device)

    train_dataloader = cfg["train_dataloader"]
    valid_dataloader = cfg["valid_dataloader"]

    train_loop(cfg, model, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    main()
