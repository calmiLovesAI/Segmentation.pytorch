from pathlib import Path

import cv2
import torch
import yaml
from typing import List


class MeanMetric:
    def __init__(self):
        self.accumulated = 0
        self.count = 0

    def update(self, value):
        self.accumulated += value
        self.count += 1

    def result(self):
        return self.accumulated / self.count

    def reset(self):
        self.__init__()


def load_yaml(filepath: List[str]) -> dict:
    cfg = dict()
    for file in filepath:
        print("Reading {}...".format(file))
        with open(file, encoding="utf-8") as f:
            # cfg |= yaml.load(f.read(), Loader=yaml.FullLoader)
            cfg.update(yaml.load(f.read(), Loader=yaml.FullLoader))
    print("Merging parsing results...")
    return cfg


def show_cfg(cfg):
    for k, v in cfg.items():
        if k not in ["model", "train_dataloader", "valid_dataloader"]:
            if isinstance(v, dict):
                print("-----------------------------")
                print(f"{k}: ")
                show_cfg(v)
                print("-----------------------------")
            else:
                print(f"{k}: {v}")


def cv2_read_image(image_path):
    image_array = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)  # (H, W, C(R, G, B)) (0~255) dtype = np.uint8
    return image_array


class Saver:
    def __init__(self, model, optimizer, scheduler):
        self.best_score = 0.0
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def set_best_score(self, best_score):
        self.best_score = best_score

    def save_ckpt(self, epoch, save_root, filename_prefix, score, overwrite=False):
        if score > self.best_score:
            if overwrite:
                # Delete the model file of current best_score.
                if self.best_score != 0.0:
                    current_best_model_path = Path(save_root).joinpath(f"{filename_prefix}_score={self.best_score}.pth")
                    current_best_model_path.unlink()
                    print(f"Remove ckpt: {current_best_model_path}")
            # save current model
            file_path = Path(save_root).joinpath(f"{filename_prefix}_score={score}.pth")
            if self.scheduler != "None":
                torch.save(obj={
                    "current_epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "best_score": score,
                }, f=file_path)
            else:
                torch.save(obj={
                    "current_epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "best_score": score,
                }, f=file_path)
            self.best_score = score
            print(f"New ckpt {file_path} saved.")
