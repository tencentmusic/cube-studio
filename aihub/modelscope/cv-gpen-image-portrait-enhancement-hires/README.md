
# 人像修复介绍

输入一张包含人像的图像，算法会对图像中的每一个检测到的人像做修复和增强，对图像中的非人像区域采用RealESRNet做两倍的超分辨率，最终返回修复后的完整图像。

## 模型描述

GPEN将预训练好的StyleGAN2网络作为decoder嵌入到人像修复模型中，并通过finetune的方式最终实现修复功能，在多项指标上达到行业领先的效果。

本模型支持1024x1024分辨率人像，能够更好处理大分辨的人像。同时模型文件列表中提供了支持2048x2048分辨率人像输入的模型，适合增强修复手机自拍的大分辨率人像，用户可以通过修改相关代码来使用。

![模型结构](description/architecture.png)

## 期望模型使用方式以及适用范围

本模型适用范围较广，给定任意的包含人像的图片，在设备性能允许的情况下，都能输出修复后的效果图。

### 如何使用

在ModelScope框架上，提供任意图片，即可以通过简单的Pipeline调用来使用人像修复模型。

#### 代码范例
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement-hires')
result = portrait_enhancement('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/marilyn_monroe_4.jpg')
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
```

### 模型局限性以及可能的偏差

- 真实图片的降质很复杂，本算法使用模拟降质数据训练，可能存在处理不好的case。
- 本算法可能存在色偏等瑕疵现象。

## 训练数据介绍

训练数据为FFHQ公开数据集。本算法采用监督式的训练，因此需要事先准备好高质-低质的数据对，推荐使用RealESRGAN、BSRGAN等降质方式进行低质数据生成。

## 模型训练流程

### 预处理

### 模型训练代码

## 数据评估及结果

| Metric | Value |
|:------:|:-----:|
|  FID   | 31.72 |
|  PSNR  | 20.80 |
|  LPIPS | 0.346 |

### 相关论文以及引用信息

```BibTeX
@inproceedings{yang2021gpen,
    title={GAN Prior Embedded Network for Blind Face Restoration in the Wild},
    author={Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
```
