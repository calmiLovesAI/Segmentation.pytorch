import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding,
                                     dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPP(nn.Module):
    def __init__(self, output_stride):
        super(ASPP, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(inplanes, 256, kernel_size=1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(inplanes, 256, kernel_size=3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(inplanes, 256, kernel_size=3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(inplanes, 256, kernel_size=3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                             nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                             nn.BatchNorm2d(num_features=256),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        self._init_weights()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp2(x)
        x4 = self.aspp2(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(input=x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat(tensors=(x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
