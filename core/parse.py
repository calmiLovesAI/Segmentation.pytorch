# Get the model and dataloader from the configuration dict

from datasets.dataloader import get_voc_dataloader
from models.fcn import FCN
from models.unet import UNet


def update_cfg(cfg: dict, opts, device, use_dataset=True) -> dict:
    cfg.update({"device": device})
    # update some training hyperparameters
    cfg["Train"]["batch_size"] = opts.batch_size
    cfg["Train"]["start_epoch"] = opts.start_epoch
    cfg["Train"]["epochs"] = opts.epochs
    cfg["Train"]["learning_rate"] = opts.lr
    cfg["Train"]["save_frequency"] = opts.save_freq
    cfg["Train"]["tensorboard_on"] = opts.tensorboard
    if cfg["Model"]["name"] == "FCN":
        fcn = FCN(num_classes=cfg["Dataset"]["num_classes"] + 1)
        cfg.update({"model": fcn})
    elif cfg["Model"]["name"] == "UNet":
        model_cfg = cfg["Model"]
        unet = UNet(num_classes=cfg["Dataset"]["num_classes"] + 1,
                    in_channels=model_cfg["Up"]["in_channels"],
                    out_channels=model_cfg["Up"]["out_channels"],
                    pretrained=model_cfg["backbone"]["pretrained"])
        cfg.update({"model": unet})
    if use_dataset:
        if cfg["Dataset"]["name"] == "VOC":
            train_dataloader = get_voc_dataloader(cfg, True)
            valid_dataloader = get_voc_dataloader(cfg, False)
            cfg.update({"train_dataloader": train_dataloader})
            cfg.update({"valid_dataloader": valid_dataloader})
    return cfg
