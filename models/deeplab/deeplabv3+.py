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
    def __init__(self, num_classes, output_stride=16, freeze_bn=False):
        super(DeeplabV3Plus, self).__init__()
        self.backbone = resnet101(output_stride, True)
        self.aspp = ASPP(output_stride)
        self.decoder = Decoder(num_classes)

        self.freeze_bn = freeze_bn

    def forward(self, x):
        size = x.size()[2:]
        x, feats = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, feats)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
