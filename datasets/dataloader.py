from typing import List

import torch
from torch.utils.data import DataLoader

from datasets.transform import ToTensor, RGB2idx, Compose, RandomCrop, Resize
from datasets.voc import VOCSegmentation, VOC_COLORMAP


def colormap2label(colormap):
    """
    构建从RGB到类别索引的映射
    :param colormap: list
    :return:
    """
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, color in enumerate(colormap):
        colormap2label[(color[0] * 256 + color[1]) * 256 + color[2]] = i
    return colormap2label


def get_voc_dataloader(cfg, is_train=True):
    batch_size = cfg["Train"]["batch_size"]
    root = cfg["Dataset"]["root"]
    base_size = cfg["Train"]["input_size"][1:]
    if isinstance(base_size, List):
        base_size = max(base_size)
        assert isinstance(base_size, int)
    crop_size = cfg["Train"]["crop_size"]
    voc_colormap2label = colormap2label(VOC_COLORMAP)
    if is_train:
        dataset = VOCSegmentation(root=root,
                                  image_set="train",
                                  transform=Compose([
                                      ToTensor(),
                                      RGB2idx(voc_colormap2label),
                                      Resize(size=base_size),
                                      RandomCrop(*crop_size),
                                  ]))
        print(f"Loading train dataset with {len(dataset)} samples")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = VOCSegmentation(root=root,
                                  image_set="val",
                                  transform=Compose([
                                      ToTensor(),
                                      RGB2idx(voc_colormap2label),
                                      Resize(size=[crop_size, crop_size])
                                  ]))
        print(f"Loading val dataset with {len(dataset)} samples")
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
