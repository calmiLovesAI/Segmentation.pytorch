# import numpy as np
# from PIL import Image
#
#
#
# img = Image.open("data/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg").convert("RGB")
# print(img.mode)
#
# tar = Image.open("data/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png").convert("RGB")
# print(tar)
# tar_numpy = np.asarray(tar)
# print(tar_numpy.shape)
import os.path

a = os.path.join("data/VOCdevkit/VOC2012/ImageSets/Segmentation", "train.txt")
with open(a) as f:
    file_names = [x.strip() for x in f.readlines()]
print(file_names)