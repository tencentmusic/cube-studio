
# DDColor 图像上色模型

该模型为黑白图像上色模型，输入一张黑白图像，实现端到端的全图上色，返回上色处理后的彩色图像。

## 模型描述

DDColor 是最新的图像上色算法，能够对输入的黑白图像生成自然生动的彩色结果。

算法整体流程如下图，使用 UNet 结构的骨干网络和图像解码器分别实现图像特征提取和特征图上采样，并利用 Transformer 结构的颜色解码器完成基于视觉语义的颜色查询，最终聚合输出彩色通道预测结果。

![ofa-image-caption](https://modelscope.cn/api/v1/models/damo/cv_ddcolor_image-colorization/repo?Revision=master&FilePath=./resources/ddcolor_arch.jpg&View=true)

## 模型期望使用方式和适用范围

该模型适用于多种格式的图像输入，给定黑白图像，生成上色后的彩色图像；给定彩色图像，将自动提取灰度通道作为输入，生成重上色的图像。

### 如何使用

在 ModelScope 框架上，提供输入图片，即可以通过简单的 Pipeline 调用来使用图像上色模型。

#### 代码范例

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_colorization = pipeline(Tasks.image_colorization, 
                       model='damo/cv_ddcolor_image-colorization')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/audrey_hepburn.jpg'
result = img_colorization(img_path)
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
```

### 模型局限性以及可能的偏差

- 本算法模型使用自然图像数据集进行训练，对于分布外场景（例如漫画等）可能产生不恰当的上色结果；
- 对于低分辨率或包含明显噪声的图像，算法可能无法得到理想的生成效果。

## 训练数据介绍

模型使用公开数据集 [ImageNet](https://www.image-net.org/) 训练，其训练集包含 128 万张自然图像。

## 数据评估及结果

本算法主要在 [ImageNet](https://www.image-net.org/) 和 [COCO-Stuff](https://github.com/nightrome/cocostuff)上测试。

| Val Name          | FID  | Colorfulness |
|:-----------------:|:----:|:------------:|
| ImageNet (val50k) | 3.92 | 38.26        |
| ImageNet (val5k)  | 0.96 | 38.65        |
| COCO-Stuff        | 5.18 | 38.48        |

## 引用

如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：

```
@article{kang2022ddcolor,
  title={DDColor: Towards Photo-Realistic and Semantic-Aware Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
  journal={arXiv preprint arXiv:2212.11613},
  year={2022}
}
```