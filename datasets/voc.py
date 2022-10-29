import os.path

from torch.utils.data import Dataset

from utils.read_image import cv2_read_image

# PASCAL VOC数据集中每种类别对应的颜色的RGB值
VOC_TABLE = {
    "background": (0, 0, 0),
    "aeroplane": (128, 0, 0),
    "bicycle": (0, 128, 0),
    "bird": (128, 128, 0),
    "boat": (0, 0, 128),
    "bottle": (128, 0, 128),
    "bus": (0, 128, 128),
    "car": (128, 128, 128),
    "cat": (64, 0, 0),
    "chair": (192, 0, 0),
    "cow": (64, 128, 0),
    "dining table": (192, 128, 0),
    "dog": (64, 0, 128),
    "horse": (192, 0, 128),
    "motorbike": (64, 128, 128),
    "person": (192, 128, 128),
    "potted plant": (0, 64, 0),
    "sheep": (128, 64, 0),
    "sofa": (0, 192, 0),
    "train": (128, 192, 0),
    "tv/monitor": (0, 64, 128)
}

VOC_COLORMAP = [v for v in VOC_TABLE.values()]

VOC_CLASSES = [k for k in VOC_TABLE.keys()]


class VOCSegmentation(Dataset):
    def __init__(self, root, image_set, transform=None):
        """
        VOC2012语义分割数据集
        :param root: (string) – Root directory of the VOC Dataset.
        :param image_set: (string, optional) – Select the image_set to use, "train", "trainval" or "val". If year=="2007", can also be "test".
        :param transform:
        """
        super(VOCSegmentation, self).__init__()
        self.transform = transform

        voc_root = os.path.join(root, "VOCdevkit", "VOC2012")
        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found.")
        splits_dir = os.path.join(voc_root, "ImageSets", "Segmentation")
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")   # train.txt, trainval.txt, val.txt
        with open(split_f) as f:
            file_names = [x.strip() for x in f.readlines()]
        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, "SegmentationClass")
        self.targets = [os.path.join(target_dir, x + ".png") for x in file_names]
        assert len(self.images) == len(self.targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = cv2_read_image(self.images[item])
        target = cv2_read_image(self.targets[item])

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target
