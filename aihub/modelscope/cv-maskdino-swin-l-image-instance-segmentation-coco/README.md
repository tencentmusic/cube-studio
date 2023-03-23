
# MaskDINO-Swin实例分割模型介绍

MaskDINO检测和分割通用框架，backbone选用Swin transformer模型。

## 模型描述

Swin transformer是一种具有金字塔结构的transformer架构，其表征通过shifted windows计算。Shifted windows方案将自注意力的计算限制在不重叠的局部窗口上，同时还允许跨窗口连接，从而带来更高的计算效率。分层的金字塔架构则让其具有在各种尺度上建模的灵活性。这些特性使swin transformer与广泛的视觉任务兼容，并在密集预测任务如COCO实例分割上达到SOTA性能。其结构如下图所示。

![Swin模型结构](description/teaser.png)

MaskDINO是一种用于目标检测、全景、实例和语义分割的统一架构。它可以实现检测和分割之间的任务和数据协同，并在相同设置下达到最先进性能。其结构示意图如下。

![MaskDINO模型结构](https://modelscope.cn/api/v1/models/damo/cv_maskdino-swin-l_image-instance-segmentation_coco/repo?Revision=master&FilePath=description/maskdino.jpg&View=true)

## 期望模型使用方式以及适用范围

本模型适用范围较广，能对图片中包含的大部分感兴趣物体（COCO 80类）进行识别和分割。

### 如何使用

在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_instance_segmentation.jpg'
output = './result.jpg'
segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_maskdino-swin-l_image-instance-segmentation_coco')
result = segmentation_pipeline(input_img)

# if you want to show the result, you can run
from modelscope.preprocessors.image import LoadImage
from modelscope.models.cv.image_instance_segmentation.postprocess_utils import show_result

numpy_image = LoadImage.convert_to_ndarray(input_img)[:, :, ::-1]   # in bgr order
show_result(numpy_image, result, out_file=output, show_box=True, show_label=True, show_score=False)

from PIL import Image
Image.open(output).show()
```

### 模型局限性以及可能的偏差

- 部分非常规图片或感兴趣物体占比太小或遮挡严重可能会影响分割结果
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试
- 本模型依赖mmcv-full，需用户在机器上正确安装[mmcv-full](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)

## 训练数据介绍

- [COCO2017](https://cocodataset.org/#detection-2017)：COCO全称是Common Objects in Context，是Microsoft团队提供的一个可以用来图像识别、检测和分割的数据集。COCO2017包含训练集118287张、验证集5000张、测试集40670张，共有80类物体。


## 模型训练流程

- 请参考原论文

### 预处理

测试时主要的预处理如下：
- Resize：先将原始图片的短边Resize到800，等比例缩放。此时如果长边超过了1333，则按照最长边为1333，重新计算Resize的scale进行Resize
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数

## 数据评估及结果

模型在COCO2017val上进行测试，结果如下:

| Backbone |   Pretrain   | Epochs | box mAP | mask mAP | #params | FLOPs | 
|:--------:|:------------:|:------:|:-------:|:--------:|:-------:|:-----:|
|  Swin-L  | ImageNet-21k |   50   |  59.0   |   52.3   |  223M   | 1326G |

可视化结果：

![source](https://modelscope.cn/api/v1/models/damo/cv_maskdino-swin-l_image-instance-segmentation_coco/repo?Revision=master&FilePath=description/demo.jpg&View=true)  ![result](https://modelscope.cn/api/v1/models/damo/cv_maskdino-swin-l_image-instance-segmentation_coco/repo?Revision=master&FilePath=description/result.jpg&View=true)

## 引用
如果你觉得该模型对你有所帮助，请考虑引用下面的相关论文：

```BibTeX
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
```BibTeX
@misc{li2022mask,
      title={Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation}, 
      author={Feng Li and Hao Zhang and Huaizhe xu and Shilong Liu and Lei Zhang and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2206.02777},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
