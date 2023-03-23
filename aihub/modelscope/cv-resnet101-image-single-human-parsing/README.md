
# M2FP单人人体解析模型介绍

## 模型描述

M2FP（Mask2Former for Parsing，[官方代码](https://github.com/soeaver/M2FP)）基于 Mask2Former 架构，并进行了一些改进以适应人体解析。 M2FP 可以适应几乎所有人体解析任务并产生惊人的性能。

模型整体架构图如下：

![M2FP模型结构](description/m2fp_arch.png)

## 期望模型使用方式以及适用范围

本模型适用于仅包含单个人体的图像，对图片中的人体各组件进行解析、分割。

### 如何使用

在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_body_reshaping.jpg'
segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-single-human-parsing')
result = segmentation_pipeline(input_img)
print(result[OutputKeys.LABELS])
```

### 模型局限性以及可能的偏差

- 部分非常规图片或人体占比太小或遮挡严重可能会影响分割结果
- 本模型依赖mmcv-full，需用户在机器上正确安装[mmcv-full](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
- 部分代码依赖CC BY-NC 4.0 license，不支持商用

## 训练数据介绍

- LIP，大规模单人人体解析数据集，[官方地址](https://lip.sysuhcp.com/index.php)。

## 模型训练流程

- 请参考原论文

## 数据评估及结果

模型在LIP验证集上进行评估，结果如下:

|  Backbone  |  Pretrain   | Epochs | mIoU  | #params | FLOPs | 
|:----------:|:-----------:|:------:|:-----:|:-------:|:-----:|
| ResNet-101 | ImageNet-1k |  150   | 59.86 |  63.0M  | 70.7G |

注：测试时使用了 TTA（Test Time Augmentation）数据增强

可视化结果：

![source](https://modelscope.cn/api/v1/models/damo/cv_resnet101_image-single-human-parsing/repo?Revision=master&FilePath=description/demo.jpg&View=true)  ![result](https://modelscope.cn/api/v1/models/damo/cv_resnet101_image-single-human-parsing/repo?Revision=master&FilePath=description/demo_result.jpg&View=true)

## 引用
如果你觉得该模型对你有所帮助，请考虑引用下面的相关论文：

```BibTeX
@article{yang2023humanparsing,
  title={Deep Learning Technique for Human Parsing: A Survey and Outlook},
  author={Lu Yang and Wenhe Jia and Shan Li and Qing Song},
  journal={arXiv preprint arXiv:2301.00394},
  year={2023}
}
```
```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2021}
}
```
```BibTeX
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```
