# Semantic-segmentation

| 名称  | 时间 |                               亮点                            | paper链接 | code链接 |
| ----- | ---- | -------------------------------------------------------- | ------------------------------- | -------- |
| FCN   | 2015 | 1. 首个端对端的针对像素级预测的全卷积网络<br /> 2.FCN是用深度神经网络来做语义分割的奠基性工作<br />  3.使用**转置卷积层来替换CNN最后的全连接层**，从而实现每个像素的预测 | [parper-FCN](https://arxiv.org/abs/1411.4038) |  [code-FCN](https://github.com/WYH67/Semantic-segmentation/tree/main/FCN) |
| UNet  | 2015 | 1.**U型对称结构**，左侧是卷积层，右侧是上采样层<br /> 2.采用skip connection，FCN用的是加操作（summation），U-Net用的是叠操作（concatenation） <br /> 3.通过跳跃结构融合低层次结构的细节特征和高层次结构中的语义特征，以提高分割精度| [parper-UNet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) | [code-UNet](https://github.com/WYH67/Semantic-segmentation/tree/main/UNet) |
| SegNet | 2017     |      |    [parper-SegNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7803544)                              |          |
|       |      |      |                                 |          |



## 待完成&完成（TODO）

### 框架（Architecture）

- [x]  🚌 FCN（PyTorch）
- [x] 🚌 UNet（PyTorch）
- [ ] 🚌 SegNet（PyTorch）
- [ ] 🚌 DeepLabv1（PyTorch）
- [ ] 🚌 DeepLabv2（PyTorch）
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

