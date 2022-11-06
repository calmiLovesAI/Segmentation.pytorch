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

### 1.2 Train
```commandline
python train.py --cfg experiments/fcn_voc.yaml
```

### 1.3 Evaluate
+ Evaluate on PASCAL VOC2012 val set
```commandline
python train.py --cfg experiments/fcn_voc.yaml --mode valid --ckpt outputs/FCN_VOC_weights.pth
```


### 1.4 Test
+ Test on single image or several images
```commandline
python test.py --cfg experiments/fcn_voc.yaml --ckpt outputs/FCN_VOC_weights.pth
```