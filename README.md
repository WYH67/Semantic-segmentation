# Semantic-segmentation

| 名称  | 时间 |                               亮点                            | paper链接 | code链接 |
| ----- | ---- | -------------------------------------------------------- | ------------------------------- | -------- |
| FCN   | 2015 | 1. 首个端对端的针对像素级预测的全卷积网络<br /> 2.FCN是用深度神经网络来做语义分割的奠基性工作<br />  3.使用**转置卷积层来替换CNN最后的全连接层**，从而实现每个像素的预测 | [parper-FCN](https://arxiv.org/abs/1411.4038) |  [code-FCN](https://github.com/WYH67/Semantic-segmentation/tree/main/FCN) |
| UNet  | 2015 | 1.**U型对称结构**，左侧是卷积层，右侧是上采样层<br /> 2.采用skip connection，FCN用的是加操作（summation），U-Net用的是叠操作（concatenation） <br /> 3.通过跳跃结构融合低层次结构的细节特征和高层次结构中的语义特征，以提高分割精度| [parper-UNet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) | [code-UNet](https://github.com/WYH67/Semantic-segmentation/tree/main/UNet) |
| SegNet | 2017 | 1.使用对称网络结构（编码器——解码器）<br />2.  提出一种新的上采样方法（反池化）<br />3.解码器使用在对应编码器的最大池化步骤中计算的**池化索引**来执行非线性上采样，这与反卷积相比，减少了参数量和运算量，而且消除了学习上采样的需要。   |    [parper-SegNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7803544)                              |     [code-SegNet](https://github.com/WYH67/Semantic-segmentation/blob/main/SegNet)     |
| DeepLabv1  | 2015 | 1.采用空洞卷积扩展感受野，获取更多的语境信息<br />2.采用完全连接的条件随机场(CRF)提高模型捕获细节的能力<br />  | [parper-Deeplabv1](https://arxiv.org/abs/1412.7062) | [code-Deeplabv1](https://github.com/WYH67/Semantic-segmentation/blob/main/DeepLab)          |
| DeepLabv2  | 2017 |1. **强调使用空洞卷积**。作为密集预测任务的强大工具，空洞卷积能够明确的控制 DCNN 内计算特征响应的分辨率。 既可以有效的扩大感受野，在不增加参数量和计算量的同时获取更多的上下文<br />2.提出了**空洞空间卷积池化金字塔**（atrous spatial pyramid pooling ( ASPP)），以多尺度的信息得到更强健的分割结果。ASPP并行的采用了多个采样率的空洞卷积层来预测，用多个比例捕获对象以及图像上下文<br /> |[parper-Deeplabv2](https://arxiv.org/pdf/1606.00915.pdf) |[code-Deeplabv2](https://github.com/WYH67/Semantic-segmentation/blob/main/DeepLab) |
| DeepLabv3  |      |      |                                 |          |

## 待完成&完成（TODO）

### 框架（Architecture）

- [x]  🚌 FCN（PyTorch）
- [x] 🚌 UNet（PyTorch）
- [x] 🚌 SegNet（PyTorch）
- [x] 🚌 DeepLabv1（PyTorch）
- [x] 🚌 DeepLabv2（PyTorch）
- [ ] 🚌 DeepLabv3（PyTorch）
- [ ] 🚌 DeepLabv3+（PyTorch）
- [ ] 🚌 RefineNet（PyTorch）
- [ ] 🚌 PSPNet（PyTorch）
- [ ] 🚌 HRNet（PyTorch）



### 组件（Components）& 模块（module）

- [ ] 🚚



### 数据增强技巧（Data Augmentation）

- [ ] 🚕



### 损失函数（Loss function）

- [ ] 🚗 Cross Entropy Loss Function

