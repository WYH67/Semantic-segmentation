from doctest import master

import torch
from torch import nn
from torchvision import models

from utils import get_upsampling_weight
from config import vgg16_caffe_path

class FCN32VGG(nn.Module):
    def __init__(self, num_classes, pretrained=True):   # 需要用预训练模型，设置pretrained=True
        super(FCN32VGG, self).__init__() # 把FCN32VGG父类的__init__()放到自己的__init__()当中，使其拥有父类中的__init__()中的东西
        vgg = models.vgg16()    # 载入vgg16模型
        if pretrained:  # 判断是否需要预训练
            vgg.load_state_dict(torch.load(vgg16_caffe_path))   # 加载预训练的参数权重

        # vgg.features是取出vgg16网络中的features大层。其中vgg网络可以分为3大层，一层是（features），一层是（avgpool），最后一层是（classifier）
        # 左边features, classifier大层被赋值，右边将feature,classifier每个子模块取出转为列表
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)    # FCN原论文中backbone的第一个卷积层padding=100，为了防止图片过小（例如192192）后面的卷积层会报错

        for f in features:
            if 'MaxPool' in f.__class__.__name__:   # 如果'MaxPool'在当前类名称中
                f.ceil_mode = True  # ceil_mode直接影响着计算方式和输出结果
            # ceil_mode 为True时的情况
            # torch.Size([1, 1, 5, 5])
            # tensor([[[[2., 3.],
            #           [5., 1.]]]])

            # ceil_mode 为False时的情况
            # torch.Size([1, 1, 5, 5])
            # tensor([[[[2.]]]])

            elif 'ReLU' in f.__class__.__name__:    # 如果'ReLU'在当前类名称中
                f.inplace = True    # 不创建新的对象，直接对原始对象进行修改

        self.features5 = nn.Sequential(*features)  # 将每一个feature按照顺序送入到nn.Sequential中

        # 之后经过FC6层：由于将FC6卷积层的padding设置为3、卷积核大小7*7，通过FC6之后将不会改变特征图的高和宽；且我们使用了4096个卷积核，所以这里就得到了4096个2D特征图。
        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))  # embedding.weight.data.copy_(weight_matrix):使用预训练的词向量，在此处指定预训练的权重
        fc6.bias.data.copy_(classifier[0].bias.data)

        # 经过FC7：使用了1*1大小的卷积核，步距也为1，所以输出特征图shape也不会发生变化
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)

        # 经过卷积核大小为1*1的卷积层：它的卷积核的个数和我们的分类类别数一样（包含背景，对于voc为20类+1背景），将特征图通道数变为num_cls
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()  # 权重设置为0
        score_fr.bias.data.zero_()  # 偏置设置为0
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )  # 将FC6和FC7的三步+1×1卷积一并顺序传入到self.score_fr

        # 通过一个转置卷积：这里的s32我们会将特征图上采样32倍[原论文中使用的是双线性插值]，得到特征图大小变为h，w，num_cls
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        # self.upscore.weight.data.copy_(get_upsampling_weight())

    def forward(self, x):
        x_size = x.size()
        pool5 = self.features5(x)
        score_fr = self.score_fr(pool5)
        upscore = self.upscore(score_fr)
        return upscore[:, :, 19: (19 + x_size[2]), 19: (19 + x_size[3])].contiguous()

if __name__ == '__main__':
    X = torch.rand(1,3,224,224)
    net = FCN32VGG(num_classes=21, pretrained=False)
    out = net(X)
    print(out.shape)
