import torch
import torch.nn as nn

'''
    DeepLabV1的模型结构十分简单，基于VGG16做了一些改动。将VGG16的最后三层全连接层替换为卷积层。
    其中，第十一、十二、十三层为dilation=2的空洞卷积，第十四层为dilation=4的空洞卷积。
'''

# 输入224 * 224 * 3
class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.stage_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 112 * 112 * 64
        self.stage_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 56 * 56 * 128
        self.stage_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 前三次卷积之后就是原图大小的1/8，后面两层卷积+池化，池化层就按步长(Stride)为1进行处理了。最终的结果就是原图的1/8。
        # 28 * 28 * 256
        self.stage_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1, padding=1),  # stride=1,和原vgg区别
        )

        # 28 * 28 * 512
        self.stage_5 = nn.Sequential(
            # 空洞卷积
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 28 * 28 * 512
            nn.MaxPool2d(2, stride=1),     # stride=1,和原vgg区别
        )
       #  Fc6，Fc7，Fc8这三个全连接层去掉。直接用池化层特征图做最后的featuremap即可
       # 最后加了一次升采样。这样就获得了与原图大小一致的特征图

    def forward(self, x):
        x = x.float()
        x1 = self.stage_1(x)
        x2 = self.stage_2(x1)
        x3 = self.stage_3(x2)
        x4 = self.stage_4(x3)
        x5 = self.stage_5(x4)
        return [x1, x2, x3, x4, x5]


class DeepLabV1(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV1, self).__init__()
        # 前13层是VGG16的前13层,分为5个stage
        self.num_classes = num_classes
        self.backbone = VGG13()

        self.stage_1 = nn.Sequential(
            # 空洞卷积
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Conv2d(512, self.num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 调用VGG16的前13层 VGG13
        x = self.backbone(x)[-1]
        x = self.stage_1(x)
        x = nn.functional.interpolate(input=x, scale_factor=8, mode='bilinear') # 上采样，获得了与原图大小一致的特征图
        x = self.final(x)
        return x

if __name__ == "__main__":
    model = DeepLabV1(21)
    image = torch.randn(1, 3, 224, 224)
    #  print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
