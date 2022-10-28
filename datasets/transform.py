import numpy as np
import torch
import torchvision.transforms.functional as F

from datasets.voc import VOC_COLORMAP


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


def label_indices(colormap, colormap2label):
    """
    将RGB值映射到对应的索引类别
    :param colormap:
    :param colormap2label:
    :return:
    """
    colormap = np.array(colormap, dtype=np.int64)  # to numpy.ndarray (h x w x c)
    colormap = torch.from_numpy(colormap)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


class ToTensor:
    def __call__(self, image):
        return F.to_tensor(image)


class RGB2idx:
    def __call__(self, target):
        target = target.convert("RGB")
        return label_indices(target, colormap2label(VOC_COLORMAP))
