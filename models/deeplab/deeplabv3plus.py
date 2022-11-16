import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deeplab.aspp import ASPP
from models.deeplab.decoder import Decoder
from models.deeplab.resnet import resnet101


class DeeplabV3Plus(nn.Module):
    """
    Paper: https://arxiv.org/pdf/1802.02611v3.pdf
    """
    def __init__(self, num_classes, output_stride=16, backbone_pretrained=True):
        super(DeeplabV3Plus, self).__init__()
        self.backbone = resnet101(output_stride, backbone_pretrained)
        self.aspp = ASPP(output_stride)
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        size = x.size()[2:]
        x, feats = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, feats)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x
