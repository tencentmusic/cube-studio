
# 人像图片生成介绍

输入随机种子，基于StyleGAN2人像生成模型，返回高清晰(分辨率为1024x1024)的人像图片。

## 模型描述

StyleGAN是图像生成领域的代表性工作，StyleGAN2在StyleGAN的基础上，采用Weight Demodulation取代AdaIN等改进极大的减少了water droplet artifacts等，生成结果有了质的提升，甚至能达到以假乱真的程度。

## 期望模型使用方式以及适用范围

本模型适用范围较广，给定任意的随机种子，就能生成一张高清晰度的人像图片。

### 如何使用

在ModelScope框架上，提供随机种子，即可以通过简单的Pipeline调用来使用人像生成模型。

#### 代码范例

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

seed = 10
face_generation = pipeline(Tasks.face_image_generation, model='damo/cv_gan_face-image-generation')
result = face_generation(seed)
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
print('done')
```

### 模型局限性以及可能的偏差

模型基于FFHQ数据集进行训练，生成结果可能会存在与训练数据分布相关的偏差。同时因为GAN的局限性，生成的人像图片可能存在不真实的瑕疵。

## 训练数据介绍

训练数据为FFHQ公开数据集。

## 模型训练流程

### 预处理

### 训练

## 数据评估及结果

| Metric |   Value   |
|:------:|:---------:|
| fid50k | 2.84±0.03 |
| is50k  | 5.13±0.02 |

### 相关论文以及引用信息

```BibTeX
@inproceedings{karras2019stylegan2,
  title     = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author    = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
```

https://github.com/rosinality/stylegan2-pytorch
