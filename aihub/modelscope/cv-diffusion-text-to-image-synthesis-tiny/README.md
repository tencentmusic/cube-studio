
# 文本到图像生成扩散模型-中英文-通用领域-tiny

输入描述文本，基于端到端文本到图像生成模型，返回符合文本描述的2D图像。

_（注意：本模型为文到图生成tiny版本。如需更优文到图生成效果，请移步完整50亿参数大模型：[文本到图像生成扩散模型-中英文-通用领域-v1.0](https://www.modelscope.cn/models/damo/cv_diffusion_text-to-image-synthesis)）_

## 模型描述

文本到图像生成模型由文本特征提取与扩散去噪模型两个子网络组成。文本特征提取子网络为StructBert结构，扩散去噪模型为unet结构。通过StructBert提取描述文本的语义特征后，送入扩散去噪unet子网络，通过迭代去噪的过程，逐步生成复合文本描述的图像。

## 期望模型使用方式以及适用范围

本模型适用范围较广，能基于任意中文文本描述进行推理，生成图像。

### 如何使用

在ModelScope框架上，提供输入文本，即可以通过简单的Pipeline调用来使用文本到图像生成模型。

#### 代码范例

```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

text2image = pipeline(Tasks.text_to_image_synthesis, 'damo/cv_diffusion_text-to-image-synthesis_tiny')
result = text2image({'text': '中国山水画'})

cv2.imwrite('result.png', result['output_imgs'][0])
```

### 模型局限性以及可能的偏差

模型基于LAION400M公开数据与互联网图文数据进行训练，生成结果可能会存在与训练数据分布相关的偏差。

## 训练数据介绍

训练数据包括LAION400M公开数据集，以及互联网图文数据。

## 模型训练流程

### 预处理

文本截断到长度64 (有效长度62)，图像缩放到64x64进行处理。

### 训练

模型分为文本特征提取与扩散去噪模型两个子网络，训练也是分别进行。文本特征提取子网络StructBert使用大规模中文文本数据上预训练得到。扩散去噪模型则使用预训练StructBert提取文本特征后，与图像一同训练文本到图像生成模型。

## 数据评估及结果

暂无。

### 相关论文以及引用信息

C Saharia, et, al. "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding." 2022.

W Wang, et, al. "Structbert: Incorporating language structures into pre-training for deep language understanding." 2019.
