'''
 数据增强：
  扩充数据样本规模的一种有效方法。深度学习是基于大数据的一种方法，我们当前希望数据的规模越大、质量越高越好。
  模型才能够有着更好的泛化能力，然而实际采集数据的时候，往往很难覆盖掉全部的场景，比如：对于光照条件，在采集图像数据时，我们很难控制光线的比例，因此在训练模型的时候，就需要加入光照变化方面的数据增强。
  再有一方面就是数据的获取也需要大量的成本，如果能够自动化的生成各种训练数据，就能做到更好的开源节流。
 
 数据增强的作用：
    1.增加训练的数据量，提高模型的泛化能力
    2.增加噪声数据，提升模型的鲁棒性
'''


#-------------------------原图-------------------------------------------#
from PIL import Image
from torchvision import transforms as tfs

img = Image.open('./dog.jpg')
print('原图：')
img


# ----------------------------随机翻转（水平、上下）---------------------------------#
# class torchvision.transforms.RandomHorizontalFlip(p=0.5)
# p（float）– 翻折图片的概率。默认0.5
rh_img = tfs.RandomHorizontalFlip(1)(img)
rh_img

# class torchvision.transforms.RandomVerticalFlip(p=0.5)
# p（float）– 翻折图片的概率。默认0.5
rv_img = tfs.RandomVerticalFlip(1)(img)
rv_img


#--------------------------------随机剪裁--------------------------------------#
rv_img = transforms.RandomCrop([200, 200])(im)
rv_img


# -----------------------------随机HSV------------------------------
'''
  为什么要RGB转HSV：
    首先RGB只是用于形成我们想要的颜色，比如说，我们想要黄色，可以通过三原色形成黄色，不管是明黄还是淡黄，只需要用不同比例进行混合就能得到我们想要的颜色.
    但是在我们进行编程的过程中不能直接用这个比例 ，需要辅助工具，也就是HSV，所以需要将RGB转化成HSV。
    HSV用更加直观的数据描述我们需要的颜色，H代表色彩，S代表深浅，V代表明暗。HSV在进行色彩分割时作用比较大。通过阈值的划分，颜色能够被区分出来
'''
rgb_point = np.array([186, 179, 151], dtype=np.uint8).reshape((1, 1, 3))
hsv_point = cv2.cvtColor(rgb_point, cv2.COLOR_RGB2HSV)
print(hsv_point)  # [[[ 24  48 186]]]



# -------------------------channel shuffling------------------------------------#
# ShuffleNet 中引入了 channel shuffle, 用来进行不同分组的特征之间的信息流动, 以提高性能。
def channel_shuffle(x, groups):

    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

# ----------------------------高斯模糊---------------------------------#
# 使用高斯核对图像进行高斯模糊变换。这种方法有助于降低图像的清晰度和清晰度，然后将生成的图像输入到神经网络中，神经网络在样本的学习模式方面变得更加稳健。
# sigma:标准差(min,max)
blurred_imgs = [T.GaussianBlur(kernel_size=(3, 3), sigma=sigma)(orig_img) for sigma in (3,7)]
plot(blurred_imgs)


# -----------------------------高斯噪声----------------------------------#
# 高斯噪声是一种常用的向整个数据集添加噪声的方法，它迫使模型学习数据中包含的最重要信息。
# 它包括注入高斯噪声矩阵，高斯噪声矩阵是从高斯分布中提取的随机值矩阵。稍后，我们将在0和1之间剪裁样本。噪声因子越高，图像的噪声越大。
#添加高斯噪声
def gaussian_noise(img,mean,sigma):
    '''
    此函数将产生高斯噪声加到图片上
    :param img:原图
    :param mean:均值
    :param sigma:标准差
    :return:噪声处理后的图片
    '''

    img = img/255  #图片灰度标准化

    noise = np.random.normal(mean, sigma, img.shape) #产生高斯噪声
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out# 这里也会返回噪声，注意返回值


# ----------------------------随机改变图片的亮度、对比度和饱和度---------------------------------#
# class torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
# brightness（float或 float类型元组(min, max)）– 亮度的扰动幅度。应当是非负数
# contrast（float或 float类型元组(min, max)）– 对比度扰动幅度。应当是非负数
# saturation（float或 float类型元组(min, max)）– 饱和度扰动幅度。应当是非负数
# hue（float或 float类型元组(min, max)）– 色相扰动幅度。hue_factor从[-hue, hue]中随机采样产生，其值应当满足0<= hue <= 0.5或-0.5 <= min <= max <= 0.5
cj_img = tfs.ColorJitter(0.8, 0.8, 0.5)(img)
cj_img


# -------------------------直方图均衡化------------------------------------#
'''
  直方图均衡也称直方图拉伸，是一种简单有效的图像增强技术，通过改变图像的直方图分布，来改变图像中各像素的灰度，主要用于增强动态范围偏小的图像的对比度。
  原始图像由于其灰度分布可能集中在较窄的区间，造成图像不够清晰（如上图左），曝光不足将使图像灰度级集中在低亮度范围内。
  采用直方图均衡化，可以把原始图像的直方图变换为均匀分布的形式，这样就增加了像素之间灰度值差别的动态范围，从而达到增强图像整体对比度的效果。
'''
def hist_gray(img_gary):
    h,w = img_gary.shape
    gray_level2 = pix_gray(img_gray)
    lut = np.zeros(256)
    for i in range(256):
        lut[i] = 255.0/(h*w)*gray_level2[i] #得到新的灰度级
    lut = np.uint8(lut + 0.5)
    out = cv2.LUT(img_gray,lut)
    return 
