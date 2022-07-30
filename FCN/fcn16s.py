# reference from import torchvision.models.segmentation.fcn
# reference from https://github.com/wkentaro/pytorch-fcn/blob/main/torchfcn/models/fcn8s.py

import torch
from torch import nn
from torchvision import models

'''
相同点：FC6、FC7、Conv2d核32s的一样
不同点：
    1.转置卷积上采样率变为了2倍，之后高和宽变为1/16
    2.下面分支经过maxpool4之后变也为1/16，通道数为512；后接上了一个1*1卷积、卷积核数量为num_cls、步长为1，得到特征图大小1/16、通道数变为num_cls
    3.之后进行一个相加操作，转置卷积上采样16倍就得到了原图大小h，w，num_cls 
'''
class FCN16VGG(nn.Module):
    def __init__(self, num_classes):
        super(FCN16VGG, self).__init__()
        vgg = models.vgg16()

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features4 = nn.Sequential(*features[: 24])  # 16+3+3+1+1
        self.features5 = nn.Sequential(*features[24:])

        # maxpool4
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)  # 1*1卷积
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        # FC6
        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)

        # FC7
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)

        # 1*1卷积
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        # 转置卷积
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)  # 合并前的4×4转置卷积
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, bias=False)  # 合并后的16×16转置卷积
        # self.upscore2.weight.data.copy_(get_upsampling_weight())
        # self.upscore16.weight.data.copy_(get_upsampling_weight())

    def forward(self, x):
        x_size = x.size()
        pool4 = self.features4(x)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore16 = self.upscore16(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                   + upscore2)
        return upscore16[:, :, 27: (27 + x_size[2]), 27: (27 + x_size[3])].contiguous()

if __name__ == '__main__':
    X = torch.rand(1,3,224,224)
    net = FCN16VGG(num_classes=21)
    out = net(X)
    print(out.shape)
