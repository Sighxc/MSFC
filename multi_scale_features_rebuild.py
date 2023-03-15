import torch.nn as nn
import torch.nn.functional as F
import torch


class SizeAlign(nn.Module):
    def __init__(self, mode='area'):
        super(SizeAlign, self).__init__()
        self.mode = mode

    def forward(self, x, ref):
        # 获取参考特征图的尺寸
        _, _, H_ref, W_ref = ref.size()

        # 如果x的尺寸与参考特征图相同，则直接返回x
        if x.size()[2:] == ref.size()[2:]:
            return x

        # 否则进行上采样或下采样
        else:
            return F.interpolate(x, size=ref.size()[2:], mode=self.mode)


class MSFRModule(nn.Module):
    def __init__(self, in_channels):
        super(MSFRModule, self).__init__()

        self.size_align = SizeAlign()

        # 定义卷积层
        self.conv3x3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, Features):
        # 对每个特征图进行SizeAlign操作，并进行卷积和加法操作
        p2 = self.size_align(Features, Features.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3))
        p3 = self.conv3x3(self.size_align(p2, Features.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3))) + \
             self.size_align(Features, Features.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3))
        p4 = self.conv3x3(self.size_align(p3, Features.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3))) + \
             self.size_align(Features, Features.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3))
        p5 = self.conv3x3(self.size_align(p4, Features)) + self.size_align(Features, Features)
        p6 = F.max_pool2d(p5, kernel_size=2, stride=2)

        # 返回重构后的特征金字塔
        return [p2, p3, p4, p5, p6]


msfr_module = MSFRModule(in_channels=256)
msfr_input = torch.load("out.pt")
msfr_output = msfr_module(msfr_input)
torch.save(msfr_output, "msfr_output.pt")









class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.msfr = MSFRModule(in_channels=64)
        self.conv2 = nn.Conv2d(64, 10, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        features = [x]
        for i in range(4):
            x = F.relu(F.max_pool2d(x, kernel_size=2, stride=2))
            features.append(x)
        features = self.msfr(features, x)
        x = features[-1]
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = self.conv2(x)
        x = x.view(-1, 10)
        return x


# 实例化网络对象
net = MyNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# 进行训练
for epoch in range(10):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
