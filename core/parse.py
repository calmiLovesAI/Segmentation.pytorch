# Get the model and dataloader from the configuration dict

from datasets.dataloader import get_voc_dataloader
from models.fcn import FCN
from models.unet import UNet
from models.deeplab.deeplabv3plus import DeeplabV3Plus


def get_model(cfg: dict, model_name: str):
    match model_name:
        case "FCN":
            return FCN(num_classes=cfg["Dataset"]["num_classes"])
        case "UNet":
            model_cfg = cfg["Model"]
            return UNet(num_classes=cfg["Dataset"]["num_classes"],
                        in_channels=model_cfg["Up"]["in_channels"],
                        out_channels=model_cfg["Up"]["out_channels"],
                        pretrained=model_cfg["backbone"]["pretrained"])
        case "DeeplabV3+":
            return DeeplabV3Plus(num_classes=cfg["Dataset"]["num_classes"],
                                 output_stride=cfg["Model"]["output_stride"],
                                 pretrained_backbone=cfg["Model"]["backbone"]["pretrained"])


def get_dataset(cfg: dict, dataset_name: str):
    match dataset_name:
        case "VOC":
            train_dataloader = get_voc_dataloader(cfg, True)
            valid_dataloader = get_voc_dataloader(cfg, False)

    return train_dataloader, valid_dataloader


def update_hyperparams(cfg: dict, opts):
    cfg["Train"]["batch_size"] = opts.batch_size
    cfg["Train"]["start_epoch"] = opts.start_epoch
    cfg["Train"]["epochs"] = opts.epochs
    cfg["Train"]["learning_rate"] = opts.lr
    cfg["Train"]["save_frequency"] = opts.save_freq
    cfg["Train"]["tensorboard_on"] = opts.tensorboard
    cfg["Dataset"]["num_classes"] += 1


def update_cfg(cfg: dict, opts, device, use_dataset=True) -> dict:
    cfg.update({"device": device})
    if opts is not None:
        # update some training hyperparameters
        update_hyperparams(cfg, opts)
    cfg.update({"model": get_model(cfg=cfg, model_name=cfg["Model"]["name"])})
    if use_dataset:
        train_dataloader, valid_dataloader = get_dataset(cfg=cfg, dataset_name=cfg["Dataset"]["name"])
        cfg.update({"train_dataloader": train_dataloader})
        cfg.update({"valid_dataloader": valid_dataloader})
    return cfg
