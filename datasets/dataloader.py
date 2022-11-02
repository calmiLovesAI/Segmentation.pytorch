import torch
from torch.utils.data import DataLoader

from datasets.transform import ToTensor, RGB2idx, Compose, RandomCrop
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


def get_voc_dataloader(root, crop_size, is_train=True):
    image_set = "train" if is_train else "val"
    shuffle = True if is_train else False
    voc_colormap2label = colormap2label(VOC_COLORMAP)
    dataset = VOCSegmentation(root=root,
                              image_set=image_set,
                              crop_size=crop_size,
                              transform=Compose([
                                  ToTensor(),
                                  RGB2idx(voc_colormap2label),
                                  RandomCrop(*crop_size),
                              ]))
    print(f"Loading {image_set} dataset with {len(dataset)} samples")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=shuffle)
    # for i, (image, target) in enumerate(dataloader):
    #     print(f"i = {i}")
    #     print(f"image形状：{image.size()}")   # torch.Size([batch, 3, 256, 256])
    #     print(f"target形状：{target.size()}")   # torch.Size([batch, 256, 256])
    #     break
    return dataloader
