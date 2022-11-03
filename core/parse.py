# Get the model and dataloader from the configuration dict

from datasets.dataloader import get_voc_dataloader
from models.fcn import FCN


def update_cfg(cfg: dict, device, use_dataset=True) -> dict:
    cfg.update({"device": device})
    if cfg["Model"]["name"] == "FCN":
        cfg.update({"model": FCN(num_classes=cfg["Dataset"]["num_classes"] + 1)})
    if use_dataset:
        if cfg["Dataset"]["name"] == "VOC":
            train_dataloader = get_voc_dataloader(cfg, True)
            valid_dataloader = get_voc_dataloader(cfg, False)
            cfg.update({"train_dataloader": train_dataloader})
            cfg.update({"valid_dataloader": valid_dataloader})
    return cfg


