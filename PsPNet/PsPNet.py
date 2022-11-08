from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn


backbone = IntermediateLayerGetter(
    resnet101(pretrained=False, replace_stride_with_dilation=[False, True, True]),
    return_layers={'layer3': 'aux', 'layer4': 'stage4'}
)

x = torch.randn(1, 3, 224, 224).cpu()
result = backbone(x)
for k, v in result.items():
    print(k, v.shape)


# Pyramid Pooling Module
class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels

        for pool_size in pool_sizes:  # 1*1,2*2,3*3,6*6
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            # 对每一个特征图利用双线性插值上采样得到原特征图相同的size，利用双线性插值上采样得到原特征图相同的size
            ppm_out = nn.functional.interpolate(ppm(x), size=x.size()[-2:], mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts



# PSPhead-利用特征获得最终的预测结果
# 利用加强特征获得预测结果
# 1.利用一个3x3卷积对特征进行整合
# 2.利用一个1x1卷积进行通道调整，调整成Num_Classes
# 3.利用resize进行上采样使得最终输出层，宽高和输入图片一样
class PSPHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=3): # 1*1,2*2,3*3,6*6
        super(PSPHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


# 构建一个FCN分割头，用于计算辅助损失
class Aux_Head(nn.Module):
    def __init__(self, in_channels=1024, num_classes=3):
        super(Aux_Head, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.decode_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels // 2),
            nn.ReLU(),

            nn.Conv2d(self.in_channels // 2, self.in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels // 4),
            nn.ReLU(),

            nn.Conv2d(self.in_channels // 4, self.num_classes, kernel_size=3, padding=1),

        )

    def forward(self, x):
        return self.decode_head(x)



# pspnet输出两个部分，一部分用来算分割loss，一部分用来算分类loss，最后加权求和
class Pspnet(nn.Module):
    def __init__(self, num_classes, aux_loss=True):
        super(Pspnet, self).__init__()
        self.num_classes = num_classes
        self.backbone = IntermediateLayerGetter(
            resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True]),
            return_layers={'layer3': "aux", 'layer4': 'stage4'}
        )
        self.aux_loss = aux_loss
        self.decoder = PSPHEAD(in_channels=2048, out_channels=512, pool_sizes=[1, 2, 3, 6],
                               num_classes=self.num_classes)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(512, self.num_classes, kernel_size=3, padding=1),
        )
        if self.aux_loss:
            self.aux_head = Aux_Head(in_channels=1024, num_classes=self.num_classes)

    def forward(self, x):
        _, _, h, w = x.size()
        feats = self.backbone(x)
        x = self.decoder(feats["stage4"])
        x = self.cls_seg(x)
        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        # 如果需要添加辅助损失
        if self.aux_loss:
            aux_output = self.aux_head(feats['aux'])
            aux_output = nn.functional.interpolate(aux_output, size=(h, w), mode='bilinear', align_corners=True)

            return {"output": x, "aux_output": aux_output}
        return {"output": x}


if __name__ == "__main__":
    model = Pspnet(num_classes=3, aux_loss=True)
    a = torch.ones([1, 3, 224, 224])
    print(a.shape)

