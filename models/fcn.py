import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.transpose_conv = nn.ConvTranspose2d(in_channels=num_classes,
                                                 out_channels=num_classes,
                                                 kernel_size=64,
                                                 stride=32,
                                                 padding=16)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = self.transpose_conv(x)
        return x
