
# 人像抠图介绍

人像抠图对输入含有人像的图像进行处理，无需任何额外输入，实现端到端人像抠图，输出四通道人像抠图结果，如下图所示：
<div align="center">
<img src="description/human.png" width="800px">
</div>

## 抠图系列模型

| [<img src="description/human.png" width="200px">](https://modelscope.cn/models/damo/cv_unet_image-matting/summary) | [<img src="description/universal.png" width="200px">](https://modelscope.cn/models/damo/cv_unet_universal-matting/summary) | [<img src="description/video.png" width="200px">](https://modelscope.cn/models/damo/cv_effnetv2_video-human-matting/summary) |[<img src="description/sky.png" width="200px">](https://modelscope.cn/models/damo/cv_hrnetocr_skychange/summary)|
|:--:|:--:|:--:|:--:| 
| [图像人像抠图](https://modelscope.cn/models/damo/cv_unet_image-matting/summary) | [通用抠图(支持商品、动物、植物、汽车等抠图)](https://modelscope.cn/models/damo/cv_unet_universal-matting/summary) | [视频人像抠图](https://modelscope.cn/models/damo/cv_effnetv2_video-human-matting/summary) | [天空替换(一键实现魔法换天空)](https://modelscope.cn/models/damo/cv_hrnetocr_skychange/summary) |

## 模型描述

模型分为粗分割和精细抠图两个子网络，将人像抠图复杂问题拆解，先粗分割再精细化分割，两部分网络均为unet结构。粗分割网络从输入图片预测人像语义分割mask，精细分割网络基于原图和粗语义分割mask预测精细的抠图结果。

## 期望模型使用方式以及适用范围

使用方式：
- 直接推理，输入图像直接进行推理

使用范围:
- 适合含有人像的图像分割，期望图像中人像占比不要过小
- 在分辨率小于2000×2000图像上可取得期望效果

目标场景:
- 需要图像抠图的场景，如换背景等

### 如何使用

本模型基于tensorflow进行训练和推理，在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用来使用人像抠图模型。

#### 代码范例
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

portrait_matting = pipeline(Tasks.portrait_matting,model='damo/cv_unet_image-matting')
result = portrait_matting('https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-matting/1.png')
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
```

### 模型局限性以及可能的偏差
- 模型训练数据有限，部分非常规图像或者人像占比过小可能会影响抠图效果。
- 在分辨率小于2000×2000图像上可取得期望效果，分辨率过大可能出现分割后边缘有模糊

## 训练数据介绍
- 训练数据从公开数据集（COCO等）、互联网搜索人像图像，并进行标注作为训练数据
- 通过将前景粘贴到不同背景图上，生成总共约90000张训练数据

## 模型训练流程
- 粗分割网络和精分割网络分开单独训练，先训练粗分割模型，然后进行精细抠图网络训练
- 先对输入图像缩放到192×160训练粗分割网络，精分割网络的训练分辨率为768×640，训练过程中进行随机数据增强，训练学习率为1e-3

### 预处理
无需进行预处理。

## 数据评估及结果
模型在自建人像分割测试数据集上（1000张图）测试，MSE可达0.003。

## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@inproceedings{liu2020boosting,
  title={Boosting semantic human matting with coarse annotations},
  author={Liu, Jinlin and Yao, Yuan and Hou, Wendi and Cui, Miaomiao and Xie, Xuansong and Zhang, Changshui and Hua, Xian-sheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8563--8572},
  year={2020}
}
```
