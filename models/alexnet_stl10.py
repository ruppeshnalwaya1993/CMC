from __future__ import print_function

import torch
import torch.nn as nn


class MyAlexNetSTL10CMC(nn.Module):
    def __init__(self, feat_dim=64):
        super(MyAlexNetSTL10CMC, self).__init__()
        self.encoder = alexnet_stl10(feat_dim=feat_dim)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=8):
        return self.encoder(x, layer)


class alexnet_stl10(nn.Module):
    def __init__(self, feat_dim=64):
        super(alexnet_stl10, self).__init__()

        self.l_to_ab = alexnet_stl10_half(in_channel=1, feat_dim=feat_dim)
        self.ab_to_l = alexnet_stl10_half(in_channel=2, feat_dim=feat_dim)

    def forward(self, x, layer=8):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab


class alexnet_stl10_half(nn.Module):
    def __init__(self, in_channel=1, feat_dim=64):
        super(alexnet_stl10_half, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(96//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96//2, 192//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(192//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(192//2, 384//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384//2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384//2, 384//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384//2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384//2, 192//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(192//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(192 * 7 * 7 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096 // 2, feat_dim)
        )
        self.l2norm = Normalize(2)

    def forward(self, x, layer):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        if layer == 5:
            return x
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x
        x = self.fc8(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


if __name__ == '__main__':

    import torch
    model = alexnet_stl10().cuda()
    data = torch.rand(10, 3, 64, 64).cuda()
    out = model(data, 5)

    for i in range(10):
        out = model(data, i)
        print(i, out[0].shape, out[1].shape)
