import numpy as np
import torch
import torchvision.transforms
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
    colormap = torch.from_numpy(colormap.astype(np.int64))
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"     {t}"
        format_string += "\n"
        return format_string


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target

    def __repr__(self)->str:
        return f"{self.__class__.__name__}()"


class RGB2idx:
    def __call__(self, image, target):
        return image, label_indices(target, colormap2label(VOC_COLORMAP))

    def __repr__(self)->str:
        return f"{self.__class__.__name__}()"


class RandomCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, target):
        rect = torchvision.transforms.RandomCrop.get_params(
            image, (self.height, self.width)
        )
        image = torchvision.transforms.functional.crop(image, *rect)
        target = torchvision.transforms.functional.crop(target, *rect)
        return image, target

    def __repr__(self):
        return f"{self.__class__.__name__}(height={self.height}, width={self.width})"
