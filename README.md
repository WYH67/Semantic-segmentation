# Semantic-segmentation

| 名称  | 时间 |                               亮点                            | paper链接 | code链接 |
| ----- | ---- | -------------------------------------------------------- | ------------------------------- | -------- |
| FCN   | 2015 | 1. 首个端对端的针对像素级预测的全卷积网络<br /> 2.FCN是用深度神经网络来做语义分割的奠基性工作<br />  3.使用**转置卷积层来替换CNN最后的全连接层**，从而实现每个像素的预测 | [paper-FCN](https://arxiv.org/abs/1411.4038) |  [code-FCN](https://github.com/WYH67/Semantic-segmentation/tree/main/FCN) |
| UNet  | 2015 | 1.**U型对称结构**，左侧是卷积层，右侧是上采样层<br /> 2.采用skip connection，FCN用的是加操作（summation），U-Net用的是叠操作（concatenation） <br /> 3.通过跳跃结构融合低层次结构的细节特征和高层次结构中的语义特征，以提高分割精度| [paper-UNet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) | [code-UNet](https://github.com/WYH67/Semantic-segmentation/tree/main/UNet) |
| SegNet | 2017 | 1.使用对称网络结构（编码器——解码器）<br />2.  提出一种新的上采样方法（反池化）<br />3.解码器使用在对应编码器的最大池化步骤中计算的**池化索引**来执行非线性上采样，这与反卷积相比，减少了参数量和运算量，而且消除了学习上采样的需要。   |    [paper-SegNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7803544)                              |     [code-SegNet](https://github.com/WYH67/Semantic-segmentation/blob/main/SegNet)     |
| DeepLabv1  | 2015 | 1.采用空洞卷积扩展感受野，获取更多的语境信息<br />2.采用完全连接的条件随机场(CRF)提高模型捕获细节的能力<br />  | [paper-Deeplabv1](https://arxiv.org/abs/1412.7062) | [code-Deeplabv1](https://github.com/WYH67/Semantic-segmentation/blob/main/DeepLab)          |
| DeepLabv2  | 2017 |1. **强调使用空洞卷积**。作为密集预测任务的强大工具，空洞卷积能够明确的控制 DCNN 内计算特征响应的分辨率。 既可以有效的扩大感受野，在不增加参数量和计算量的同时获取更多的上下文<br />2.提出了**空洞空间卷积池化金字塔**（atrous spatial pyramid pooling ( ASPP)），以多尺度的信息得到更强健的分割结果。ASPP并行的采用了多个采样率的空洞卷积层来预测，用多个比例捕获对象以及图像上下文<br /> |[paper-Deeplabv2](https://arxiv.org/pdf/1606.00915.pdf) |[code-Deeplabv2](https://github.com/WYH67/Semantic-segmentation/blob/main/DeepLab) |
| DeepLabv3  | 2017 |1.本文重新讨论了空洞卷积的使用，在串行模块和空间金字塔池化的框架下，能够获取更大的感受野从而获取多尺度信息<br />2.改进了ASPP模块：由不同采样率的空洞卷积和BN层组成，我们尝试以串行或并行的方式布局模块<br />3.讨论了一个重要问题：使用大采样率的3×3的空洞卷积，因为图像边界响应无法捕捉远距离信息(小目标)，会退化为1×1的卷积, 我们建议将图像级特征融合到ASPP模块中      | [paper-Deeplabv3](https://arxiv.org/pdf/1706.05587.pdf) | [code-Deeplabv3](https://github.com/WYH67/Semantic-segmentation/blob/main/DeepLab) |
| DeepLabv3+ | 2018 | 1.扩展了DeepLabv3，添加了一个简单而有效的**解码器模块**来细化分割结果，特别是沿着对象边界<br />2.使用DeepLabv3作为一个强大的编码器模块和一个简单而有效的解码器模块<br />3.为了在多个尺度上获取上下文信息，DeepLabv3应用了多个不同速率的并行atrous卷积<br /> | [paper-Deeplabv3+](https://arxiv.org/pdf/1802.02611.pdf) | [code-Deeplabv3+](https://github.com/WYH67/Semantic-segmentation/blob/main/DeepLab)|
| HRNet | 2019 | 1.高分辨率到低分辨率的卷积流并行连接.<br />2.跨分辨率重复交换信息。<br />好处是结果表示在语义上更丰富，在空间上更精确。 | [paper-HRNet](https://arxiv.org/abs/1902.09212) | [code-HRNet](https://github.com/WYH67/Semantic-segmentation/blob/main/HRNet/hrnet.py) |

## 待完成&完成（TODO）

### 框架（Architecture）

- [x]  🚌 FCN（PyTorch）
- [x] 🚌 UNet（PyTorch）
- [x] 🚌 SegNet（PyTorch）
- [x] 🚌 DeepLabv1（PyTorch）
- [x] 🚌 DeepLabv2（PyTorch）
- [x] 🚌 DeepLabv3（PyTorch）
- [x] 🚌 DeepLabv3+（PyTorch）
- [x] 🚌 HRNet（PyTorch）
- [ ] 🚌 RefineNet（PyTorch）
- [ ] 🚌 PSPNet（PyTorch）



### 组件（Components）& 模块（module）

- [ ] 🚚



### 数据增强技巧（Data Augmentation）

- [x] 🚕随机翻转（水平、上下）
- [x] 🚕随机HSV
- [x] 🚕channel shuffling
- [x] 🚕高斯模糊
- [x] 🚕高斯噪声
- [x] 🚕亮度变换
- [x] 🚕HUE、饱和度变换
- [x] 🚕直方图均衡化


### 损失函数（Loss function）

- [x] 🚗 L1loss
- [x] 🚗 MSELoss
- [x] 🚗 CroosEntropyLoss
- [x] 🚗 NLLLoss
- [ ] 🚗 PoissonNLLLoss
- [ ] 🚗 KLDivLoss

### 平均交并比（MIoU）
- [x] 🚜 

### 优化器(10种)
- [ ] 🚐 Optimizer

### 学习率调整⽅法(6种)
- [ ] ✈️ lr_scheduler
