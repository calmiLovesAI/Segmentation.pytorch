import random

import numpy as np
import torch
import torchvision.transforms
import torchvision.transforms.functional as F


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


class PIL2Numpy:
    def __call__(self, image, target):
        target = target.convert("RGB")
        return np.asarray(image), np.asarray(target)


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        target = F.resize(target.unsqueeze(dim=0), size=self.size)
        return F.resize(image, size=self.size), torch.squeeze(target, dim=0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RGB2idx:
    def __init__(self, colormap2label):
        self.colormap2label = colormap2label

    def __call__(self, image, target):
        return image, label_indices(target, self.colormap2label)

    def __repr__(self) -> str:
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


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label.
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        return F.normalize(image, mean=self.mean, std=self.std), target

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomHorizontalFlip:
    """Horizontally flip the given Tensor randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        """
        Args:
            image (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(image), F.hflip(target)
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
