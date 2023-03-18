
# DeOldify for Video Colorizaiton

[**English Version**](#DeOldify%20for%20Video%20Colorizaiton) **|** [**中文版本**](#DeOldify%20视频上色)

Input a grayscale video and automatically output the corresponding color video.

## Model description

DeOldify is a famous open source algorithm for automantic colorization. The model uses ResNet as encoder to build a network with UNet structure, and puts forward several different training versions, which has a good comprehensive performance in the aspects of effect, efficiency, robustness and so on.

The model here is the video model of DeOldify, which makes some optimization in reducing flicker. The video colorization process is realized using isolated image generation without any sort of temporal modeling tacked on.

**Model structure**

<img src="resources/deoldify_arch.png">

## Usage

Given a sequence of grayscale video frames, color video frames will be generated. When providing color video frames, the result frames will be recolored.

### How to use

With the ModelScope framework, you can use the colorization model through a simple Pipeline call.

#### Code example

```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video = 'https://vigen-video.oss-cn-shanghai.aliyuncs.com/ModelScope/test/videos/gray.mp4'
colorizer = pipeline(Tasks.video_colorization, model='damo/cv_unet_video-colorization')
result = colorizer(video)[OutputKeys.OUTPUT_VIDEO]
```

### Limitations and possible deviations

- This model is the video model of DeOldify, which may be less colorful.
- Due to the limitations of training data and model structure, the algorithm may have some generation defects.


## Evaluation results

This algorithm is tested on [ImageNet](https://www.image-net.org/)，[COCO-Stuff](https://github.com/nightrome/cocostuff) and [CelebA-HQ](https://www.tensorflow.org/datasets/catalog/celeb_a_hq).

| Metric | ImageNet | COCO-Stuff | CelebA-HQ |
|:------:|:--------:|:----------:|:---------:|
|  FID   |   3.87   |   13.86    |   9.48    |
|  PSNR  |   22.97  |   24.19    |   25.20   |

### Reference

```BibTeX
@misc{deoldify,
    author    = {J. Antic},
    title     = {A deep learning based project for colorizing and restoring old images (and video!)},
    howpublished = "\url{https://github.com/jantic/DeOldify}",
}
```


# DeOldify 视频上色

[**English Version**](#DeOldify%20for%20Video%20Colorizaiton) **|** [**中文版本**](#DeOldify%20视频上色)

输入一段黑白视频，全自动地输出相对应的彩色视频。

## 模型描述

DeOldify 是上色领域比较有名的开源算法，模型利用 ResNet 作为 encoder 构建一个 UNet 结构的网络，并提出了多个不同的训练版本，在效果、效率、鲁棒性等等方面有良好的综合表现。

本模型为用于视频的版本，在减少视频中的闪烁方面做出了一定优化。视频上色过程是使用对每一帧的独立上色来实现，没有附加任何类型的时间建模。

**模型结构**

<img src="resources/deoldify_arch.png">

## 期望模型使用方式以及适用范围

本模型适用范围较广，给定一段任意的视频，都能生成上色后的彩色视频。如果输入的是彩色视频，将进行重上色。

### 如何使用

在 ModelScope 框架上，提供任意视频，即可以通过简单的 Pipeline 调用来使用视频上色模型。

#### 代码范例

```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video = 'https://vigen-video.oss-cn-shanghai.aliyuncs.com/ModelScope/test/videos/gray.mp4'
colorizer = pipeline(Tasks.video_colorization, model='damo/cv_unet_video-colorization')
result = colorizer(video)[OutputKeys.OUTPUT_VIDEO]
```

### 模型局限性以及可能的偏差

- 本算法使用的是用于视频上色版本的 DeOldify 模型，在色彩鲜艳度上可能会稍逊一些。
- 因训练数据、模型结构等限制，算法存在一些生成瑕疵效果。

## 数据评估及结果

本算法主要在 [ImageNet](https://www.image-net.org/)，[COCO-Stuff](https://github.com/nightrome/cocostuff) 和 [CelebA-HQ](https://www.tensorflow.org/datasets/catalog/celeb_a_hq) 上测试。

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