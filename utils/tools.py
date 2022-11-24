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

    def save_ckpt(self, epoch, filename, score):
        if score > self.best_score:
            # save current model
            torch.save(obj={
                "current_epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_score": score,
            }, f=filename)
            self.best_score = score


