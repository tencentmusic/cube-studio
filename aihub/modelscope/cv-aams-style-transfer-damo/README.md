
# 风格迁移介绍

给定内容图像和风格图像作为输入，风格迁移模型会自动地将内容图像的风格、纹理特征变换为风格图像的类型，同时保证图像的内容特征不变

## [项目主页](https://sites.google.com/view/yuanyao/attention-aware-multi-stroke-style-transfer)

![内容图像](https://modelscope.cn/api/v1/models/damo/cv_aams_style-transfer_damo/repo?Revision=master&FilePath=description/showcase_small.jpg&View=true)


## 模型描述

本模型将视觉注意力与图像风格迁移任务结合，通过在编解码器网络中增加自注意力模块、多笔触融合模块和风格交换模块，实现了多笔触渲染控制，保证了风格迁移前后的注意力一致性

## 期望模型使用方式以及适用范围

使用方式：
- 直接推理，输入图像直接进行推理

使用范围:
- 在分辨率小于1200×1200图像上可取得期望效果

目标场景:
- 互动娱乐，图像滤镜生成等场景

### 如何使用

本模型基于tensorflow进行训练和推理，在ModelScope框架上，提供输入的内容图片和相应风格图片，即可以通过简单的Pipeline调用来使用风格迁移模型

#### 代码范例
```python

import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

content_img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-style-transfer/style_transfer_content.jpg'
style_img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-style-transfer/style_transfer_style.jpg'
style_transfer = pipeline(Tasks.image_style_transfer, model_id='damo/cv_aams_style-transfer_damo')
result = style_transfer(dict(content = content_img, style = style_img))
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])

```

### 模型局限性以及可能的偏差
- 对细节要求高的场景可能无法完全保证风格迁移结果的纹理结构稳定
- 在分辨率小于1200×1200图像上可取得期望效果，分辨率过大可能出现笔触粗糙和模糊现象

## 训练数据介绍
- COCO数据集作为训练数据


## 模型训练流程
- 采用MS-COCO数据集训练自注意力自编解码器，编码器采用在ImageNet上预训练的VGG-19网络，解码器与编码器结构对称。
- 先对输入图像缩放短边为512， 然后随机数据增强，裁剪成256×256的图像进行训练。

### 预处理
无需进行预处理。

## 数据评估及结果
模型在自建风格迁移测试数据集上（150张图）测试并进行用户调研，与AdaIN，WCT，Avatar-Net算法相比，本算法在风格化效果和内容保真度两个维度上均第一。

| Method | Stylization Effects | Faithful to Content | 
| ------------ | ------------ | ------------ |  
| AdaIN | 0.17 | 0.29 | 
| WCT | 0.29  |  0.2|
| Avatar-Net | 0.2 | 0.08 | 
| Ours | **0.36** | **0.42** |


## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@inproceedings{yao2019attention,
  title={Attention-aware multi-stroke style transfer},
  author={Yao, Yuan and Ren, Jianqiang and Xie, Xuansong and Liu, Weidong and Liu, Yong-Jin and Wang, Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1467--1475},
  year={2019}
}
```
