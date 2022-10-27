import torchvision


def get_voc_dataloader(is_train=True, transform=None, target_transform=None):
    image_set = "train" if is_train else "val"
    torchvision.datasets.VOCSegmentation(root="data",
                                         year="2012",
                                         image_set=image_set,
                                         download=False,
                                         transform=transform,
                                         target_transform=target_transform)
