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

    def save_ckpt(self, epoch, save_root, filename_prefix, score, overwrite=False):
        if score > self.best_score:
            if overwrite:
                # Delete the model file of current best_score.
                current_best_model_path = Path(save_root).joinpath(f"{filename_prefix}_score={self.best_score}.pth")
                current_best_model_path.unlink()
            # save current model
            file_path = Path(save_root).joinpath(f"{filename_prefix}_score={score}.pth")
            torch.save(obj={
                "current_epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_score": score,
            }, f=file_path)
            self.best_score = score


