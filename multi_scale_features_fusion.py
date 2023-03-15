import torch
from tenbitquantz import *
import torch.nn as nn
import torch.nn.functional as F

#定义SEBlock
class SEBlock(nn.Module):
    def __init__(self, channels, mode="avg", ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.global_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v


# 定义MSFFModule
class MSFFModule(nn.Module):
    def __init__(self, in_channels):
        super(MSFFModule, self).__init__()

        # 定义3个SEBlock，对应3个张量
        self.se1 = SEBlock(in_channels)
        self.se2 = SEBlock(in_channels)
        self.se3 = SEBlock(in_channels)
        self.se4 = SEBlock(in_channels)

        self.conv_reduce = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False
        )
    def SENet(self, features):
        # 对每个张量分别应用SEBlock
        x1 = self.se1(features[0])
        x2 = self.se2(features[1])
        x3 = self.se3(features[2])
        x4 = self.se4(features[3])

        # 将加权后的张量求和融合
        out = x1 + x2 + x3 + x4

        return out

    def forward(self,features):
        features[0] = F.avg_pool2d(features[0], kernel_size=8, stride=8)
        # x_downsampled.shape为[batch_size, channels, height // scale_factor, width // scale_factor]
        features[1] = F.avg_pool2d(features[1], kernel_size=4, stride=4)
        features[2] = F.avg_pool2d(features[2], kernel_size=2, stride=2)

        # 实例化SENet，输入通道数为256
        net = MSFFModule(in_channels=256)

        # 融合4个张量
        SEout = net.SENet(features)

        out = net.conv_reduce(SEout)
        return out



#以下为gpt实现方案
'''
class MSFFModule(nn.Module):
    def __init__(self, in_channels):
        super(MSFFModule, self).__init__()

        # 定义SE块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 定义卷积层用于信道约简
        self.conv_reduce = nn.Conv2d(
            in_channels , 256, kernel_size=1, bias=False
        )

    def forward(self, features):
        # 将特征金字塔中的特征按照尺度大小排序，从小到大
        features = sorted(features, key=lambda x: x.shape[-1])

        # 将每个特征图下采样到p5的大小
        p5 = features[0]
        features = [
            F.interpolate(feature, size=p5.shape[-2:], mode='nearest')
            for feature in features[:-1]
        ]
        features.append(p5)

        # 将下采样后的特征图拼接在一起
        features_concat = torch.cat(features, dim=1)

        # 计算SE块的权重并对特征进行加权
        se_weights = self.se(features_concat)
        features_weighted = features_concat * se_weights

        # 使用卷积层进行信道约简
        features_reduced = self.conv_reduce(features_weighted)

        return features_reduced
'''

msff_module = MSFFModule(256)
p2 = torch.load("p2.pt")
p3 = torch.load("p3.pt")
p4 = torch.load("p4.pt")
p5 = torch.load("p5.pt")
extracted_features = [p2, p3, p4, p5]  # 从特征金字塔中获取特征图
fusion_features = msff_module(extracted_features)  # 进行特征融合
torch.save(fusion_features, 'out.pt')
