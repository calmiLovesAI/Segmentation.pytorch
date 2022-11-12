import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=48)
        self.relu = nn.ReLU(inplace=True)
        self.final = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        )

        self._init_weights()

    def forward(self, x, feats):
        feats = self.relu(self.bn1(self.conv1(feats)))
        x = F.interpolate(x, size=feats.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat(tensors=(x, feats), dim=1)
        x = self.final(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
