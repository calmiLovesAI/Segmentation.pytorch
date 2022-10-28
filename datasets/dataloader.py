import torchvision
from torch.utils.data import DataLoader


def get_voc_dataloader(is_train=True, transform=None, target_transform=None):
    image_set = "train" if is_train else "val"
    shuffle = True if is_train else False
    dataset = torchvision.datasets.VOCSegmentation(root="data",
                                         year="2012",
                                         image_set=image_set,
                                         download=False,
                                         transform=transform,
                                         target_transform=target_transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=shuffle)
    for i, (image, target) in enumerate(dataloader):
        print(f"i = {i}")
        print(f"image形状：{image.size()}")
        print(f"target形状：{target.size()}")
        break
