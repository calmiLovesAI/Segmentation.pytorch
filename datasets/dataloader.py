import torchvision
from torch.utils.data import DataLoader

from datasets.transform import ToTensor, RGB2idx, Compose, RandomCrop
from datasets.voc import VOCSegmentation


def get_voc_dataloader(cfg, is_train=True):
    image_set = "train" if is_train else "val"
    shuffle = True if is_train else False
    dataset = VOCSegmentation(root="data",
                              image_set=image_set,
                              crop_size=(256, 256),
                              transform=Compose([
                                  ToTensor(),
                                  RGB2idx(),
                                  RandomCrop(256, 256),
                              ]))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=shuffle)
    # for i, (image, target) in enumerate(dataloader):
    #     print(f"i = {i}")
    #     print(f"image形状：{image.size()}")   # torch.Size([batch, 3, 256, 256])
    #     print(f"target形状：{target.size()}")   # torch.Size([batch, 256, 256])
    #     break
    return dataloader
