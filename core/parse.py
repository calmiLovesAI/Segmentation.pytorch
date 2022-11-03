# Get the model and dataloader from the configuration dict

from datasets.dataloader import get_voc_dataloader
from models.fcn import FCN
from models.unet import UNet


def update_cfg(cfg: dict, device, use_dataset=True) -> dict:
    cfg.update({"device": device})
    if cfg["Model"]["name"] == "FCN":
        cfg.update({"model": FCN(num_classes=cfg["Dataset"]["num_classes"] + 1)})
    elif cfg["Model"]["name"] == "UNet":
        model_cfg =cfg["Model"]
        cfg.update({"model": UNet(num_classes=cfg["Dataset"]["num_classes"] + 1,
                                  in_channels=model_cfg["Up"]["in_channels"],
                                  out_channels=model_cfg["Up"]["out_channels"],
                                  pretrained=model_cfg["backbone"]["pretrained"])})
    if use_dataset:
        if cfg["Dataset"]["name"] == "VOC":
            train_dataloader = get_voc_dataloader(cfg, True)
            valid_dataloader = get_voc_dataloader(cfg, False)
            cfg.update({"train_dataloader": train_dataloader})
            cfg.update({"valid_dataloader": valid_dataloader})
    return cfg


