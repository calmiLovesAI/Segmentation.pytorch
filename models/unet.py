import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class UNet(nn.Module):
    """
    Part of the code derives from https://blog.csdn.net/smallworldxyl/article/details/121409052
    """

    def __init__(self, num_classes, in_channels, out_channels, pretrained=False):
        """
        :param num_classes: (int) - number of categories for classification
        :param pretrained: (boolean) - True means to use pretrained weights on ImageNet.
        """
        super(UNet, self).__init__()
        if pretrained:
            self.backbone = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        else:
            self.backbone = vgg16_bn(weights=None)
        del self.backbone.classifier
        del self.backbone.avgpool

    def forward(self, x):
        pass


class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super(Up, self).__init__()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        output = torch.concat([x, self.upsampling(y)], dim=1)
        output = self.conv1(output)
        output = self.conv2(output)
        return output


if __name__ == '__main__':
    unet = UNet(21, [192, 384, 768, 1024], [64, 128, 256, 512], True)
