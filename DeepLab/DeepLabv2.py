import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import math

model_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'


# ResNet101 + atrous spatial pyramid pooling

# Bottleneck结构
class Atrous_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Atrous_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # Conv2d 1*1,s1
        self.bn1 = nn.BatchNorm2d(planes)                                    # BN
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, # Conv2d 3*3,s1,r,p
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)                                    # BN
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False) # Conv2d 1*1,s1
        self.bn3 = nn.BatchNorm2d(planes * 4)    # BN
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # 三者加在一起最后Relu
        out = self.relu(out)

        return out


# 输入224 * 224 *3
class Atrous_ResNet_features(nn.Module):

    def __init__(self, block, layers, pretrained=False):
        super(Atrous_ResNet_features, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, # Conv2d 7*7,s2,p3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)     # BN
        self.relu = nn.ReLU(inplace=True)  # ReLU

        # 112 * 112 * 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # MaxPool 3*3,s2,p1

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, rate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, rate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, rate=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, rate=4)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # 是否需要预训练
        if pretrained:
            print('load the pre-trained model.')
            resnet = models.resnet101(pretrained)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2


    # _make_layer()函数用来产生4个layer，可以根据输入的layers列表来创建网络
    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: # planes * block.expansion:每个blocks的剩下residual 结构
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, rate=rate))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)   # 7*7卷积
        x = self.bn1(x)     # BN
        x = self.relu(x)    # ReLU
        x = self.maxpool(x) # MaxPooling

        x = self.layer1(x)  # layer1~layer4
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

#  ASPP
class Atrous_module(nn.Module):
    def __init__(self, inplanes, num_classes, rate):
        super(Atrous_module, self).__init__()
        planes = inplanes
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.fc1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(planes, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class DeepLabv2_ASPP(nn.Module):
    def __init__(self, num_classes, small=True, pretrained=False):
        super(DeepLabv2_ASPP, self).__init__()
        block = Atrous_Bottleneck
        self.resnet_features = Atrous_ResNet_features(block, [3, 4, 23, 3], pretrained)

        # 根据输出的大小来确定rate
        if small:
            rates = [2, 4, 8, 12]
        else:
            rates = [6, 12, 18, 24]
        # 根据不同的rate得到不同的块
        self.aspp1 = Atrous_module(2048, num_classes, rate=rates[0])
        self.aspp2 = Atrous_module(2048, num_classes, rate=rates[1])
        self.aspp3 = Atrous_module(2048, num_classes, rate=rates[2])
        self.aspp4 = Atrous_module(2048, num_classes, rate=rates[3])

    def forward(self, x):
        x = self.resnet_features(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        # 28 * 28 * num_classes
        x = x1 + x2 + x3 + x4
        x = F.upsample(x, scale_factor=8, mode='bilinear')  # Upsample * 8

        # 224 * 224 * num_classes

        return x


if __name__ == "__main__":
     model = DeepLabv2_ASPP(num_classes=21, small=True, pretrained=False)
     image = torch.randn(1, 3, 224, 224)

     # print(model)
     print("input:", image.shape)
     print("output:", model(image).shape)
