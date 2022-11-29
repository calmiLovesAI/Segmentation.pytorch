# Segmentation.pytorch


## 1. Installation
### 1.1 Prepare the Datasets
Put the [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) dataset in the following path:
```
data
  |---VOCdevkit
       |---VOC2012
            |-------Annotations
            |-------Imagesets
            |-------JPEGImages
            |-------SegmentationClass
            |-------SegmentationObject
```
Put the [Semantic Boundaries Dataset](http://home.bharathh.info/pubs/codes/SBD/download.html) in the following path:
```
data
  |---SBD
       |---cls
       |---img
       |---inst
       |---train.txt
       |---val.txt
```

### 1.2 Train
+ Train on Semantic Boundaries Dataset (DeeplabV3+ for example)
```commandline
python train.py --cfg experiments/deeplabv3plus_sbd.yaml
```
During training, you can use the command
```commandline
tensorboard --logdir=runs
```
in the console to enter the tensorboard panel to visualize the training process.

### 1.3 Evaluate
+ Evaluate on Semantic Boundaries Dataset (DeeplabV3+ for example)
```commandline
python train.py --cfg experiments/deeplabv3plus_sbd.yaml --mode valid --ckpt outputs/DeeplabV3Plus_SBD_weights.pth
```


### 1.4 Test
+ Test on single image or several images (DeeplabV3+ for example)
```commandline
python test.py --cfg experiments/deeplabv3plus_sbd.yaml --ckpt outputs/DeeplabV3Plus_SBD_weights.pth
```


## 2. Results

## 3. Deployment

## Acknowledgments
+ https://github.com/VainF/DeepLabV3Plus-Pytorch