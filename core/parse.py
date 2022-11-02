# Get the model and dataloader from the configuration dict

from datasets.dataloader import get_voc_dataloader
from models.fcn import FCN


def update_cfg(cfg: dict, device) -> dict:
    cfg.update({"device": device})
    if cfg["Model"]["name"] == "FCN":
        cfg.update({"model": FCN(num_classes=cfg["Dataset"]["num_classes"] + 1)})
    if cfg["Dataset"]["name"] == "VOC":
        root = cfg["Dataset"]["root"]
        crop_size = cfg["Train"]["input_size"][1:]
        train_dataloader = get_voc_dataloader(root, crop_size, True)
        valid_dataloader = get_voc_dataloader(root, crop_size, False)
        cfg.update({"train_dataloader": train_dataloader})
        cfg.update({"valid_dataloader": valid_dataloader})
    return cfg


