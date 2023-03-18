
# 图像超分辨率介绍

输入低分辨率图片，返回4倍超分辨率后的高清晰图片。模型基于RealESRAGN中的降质方式进行训练，在复杂真实降质图片上也有较好的表现。

## 模型描述

RealESRGAN提出了通过多次降质的方式来模拟真实复杂降质，相比较于之前的简单下采样，能够更好处理真实的低分辨率场景。

**数据降质方式**

采用high order degradation的方式，具体流程如下：
<img src="description/high-order.png">

**模型结构**

采用ESRGAN的基本结构，针对不同超分倍数的输入特别处理，具体如下：
<img src="description/real-esrgan.png">

## 期望模型使用方式以及适用范围

本模型适用范围较广，给定任意的低分辨率图片，都能生成一张4倍超分辨率后的高清晰度图片。

### 如何使用

在ModelScope框架上，提供低分辨图片，即可以通过简单的Pipeline调用来使用图像超分辨率模型。

#### 代码范例

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

sr = pipeline(Tasks.image_super_resolution, model='damo/cv_rrdb_image-super-resolution')
result = sr('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/dogs.jpg')
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
```

### 模型局限性以及可能的偏差

通过多次降质的方式虽然能更好的模拟真实降质数据，但是并不能完全消除模拟数据与真实数据之间的domain gap，因此存在修复产生瑕疵的场景。同时，模型文件较大，输入图片的分辨率不宜过大，否则容易出现OOM错误。

## 训练数据介绍

训练数据为DIV2K, Flicker2K, FFHQ等公开数据集。

## 模型训练流程

### 预处理

### 训练

## 数据评估及结果

| Metric |  Set5  |  Set14 | Manga109 | General100 | Urban100 | DIV2K100 |
|:------:|:------:|:------:|:--------:|:----------:|:--------:|:--------:|
| LPIPS  | 0.0691 | 0.1132 |  0.0544  |   0.0796   |  0.1084  |  0.0999  |
| DISTS  | 0.0919 | 0.0866 |  0.0355  |   0.0801   |  0.0793  |  0.0526  |
|  FID   | 24.803 | 43.454 |  10.161  |   27.211   |  16.351  |  12.121  |

### 相关论文以及引用信息

```BibTeX
@inproceedings{liang2022LDL,
    author    = {Liang, Jie and Zeng, Hui and Zhang, Lei},
    title     = {Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    date      = {2022}
}
```

```BibTeX
@inproceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```
