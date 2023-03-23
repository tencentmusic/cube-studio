
# ECBSR端上图像超分模型

## 模型描述

输入Y通道的低分辨率图像（单通道灰度图像），返回2倍超分辨率后的高清晰Y通道图像。模型基于Edgeoriented
Convolution Block (ECB)模块构建，完整模型可导出为简洁的CNN网络，适用于移动端、嵌入式等严格限制算力的场景。为了适用于大部分移动端场景，模型只支持单通道图像处理，如果是RGB图像，需要将图像从RGB颜色空间转换为YCbCr格式并只提取Y通道部分输入给模型处理。完整模型结构如下所示：

![模型流程图](https://modelscope.cn/api/v1/models/damo/cv_ecbsr_image-super-resolution_mobile/repo?Revision=master&FilePath=assets/ecbsr_modelscope.jpg&View=true)

其中Edgeoriented Convolution Block (ECB)模块可以通过重参数化技术等价转换为一个普通的3x3卷积模块。

![ecb模块图](https://modelscope.cn/api/v1/models/damo/cv_ecbsr_image-super-resolution_mobile/repo?Revision=master&FilePath=assets/ecb_modelscope.jpg&View=true)

## 期望模型使用方式以及适用范围
本模型使用于移动端等算力限制严格的普通Y通道图像超分辨率。

### 如何使用
在ModelScope框架上，提供输入图像，即可通过简单的Pipeline调用来使用。

#### 代码范例

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

sr = pipeline(Tasks.image_super_resolution, model='damo/cv_ecbsr_image-super-resolution_mobile')
result = sr('https://vigen-video.oss-cn-shanghai.aliyuncs.com/VideoEnhancement/Dataset/ClassicalSRDataset/butterfly_lrx2_y.png')
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
```

### 模型局限性以及可能的偏差

本模型训练时使用经典的bicubic下采样从HR图像生成LR图像，以构造训练数据对，在其他严重降质场景效果可能不佳。本模型针对移动端和嵌入式芯片场景设计，在效果与算力之间作了折中。

## 训练数据介绍

训练数据为DIV2K公开数据集。


## 数据评估及结果

| Metric |  Set5  |  Set14 |  B100   |  U100   | DIV2K   |
|:------:|:------:|:------:|:-------:|:-------:|:-------:|
| PSNR   | 36.93  | 32.51  |  31.44  |  29.68  |  34.80  |
| SSIM   | 0.9577 | 0.9107 |  0.8932 |  0.9014 |  0.9356 |

### 相关论文以及引用信息

```BibTeX
@inproceedings{zhang2021edge,
  title={Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices},
  author={Zhang, Xindong and Zeng, Hui and Zhang, Lei},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={4034--4043},
  year={2021}
}
```