
# 图像上色介绍

输入一张黑白图片，全自动的输出相对应的彩色图片。

## 模型描述

DeOldify是图像上色领域比较有名的开源算法，模型利用resnet作为encoder构建一个unet结构的网络，并提出了多个不同的训练版本，在效果、效率、鲁棒性等等方面有良好的综合表现。

**模型结构**
<img src="description/deoldify_arch.png">

## 期望模型使用方式以及适用范围

本模型适用范围较广，给定任意的图片，都能生成一张上色后的彩色图片。如果输入的是彩色图片，将进行重上色。

### 如何使用

在ModelScope框架上，提供任意图片，即可以通过简单的Pipeline调用来使用图像上色模型。

#### 代码范例

```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

colorizer = pipeline(Tasks.image_colorization, model='damo/cv_unet_image-colorization')
result = colorizer('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/marilyn_monroe_4.jpg')
cv2.imwrite('result.png', result['output_img'])
```

### 模型局限性以及可能的偏差

- 本算法使用的是稳定版本的DeOldify模型，在色彩鲜艳度上可能会稍逊一些。
- 因训练数据、模型结构等限制，算法存在一些生成瑕疵效果。

## 数据评估及结果

本算法主要在[ImageNet](https://www.image-net.org/)，[COCO-Stuff](https://github.com/nightrome/cocostuff)和[CelebA-HQ](https://www.tensorflow.org/datasets/catalog/celeb_a_hq)上测试。

| Metric | ImageNet | COCO-Stuff | CelebA-HQ |
|:------:|:--------:|:----------:|:---------:|
|  FID   |   3.87   |   13.86    |   9.48    |
|  PSNR  |   22.97  |   24.19    |   25.20   |

### 相关论文以及引用信息

```BibTeX
@misc{deoldify,
    author    = {J. Antic},
    title     = {A deep learning based project for colorizing and restoring old images (and video!)},
    howpublished = "\url{https://github.com/jantic/DeOldify}",
}
```
