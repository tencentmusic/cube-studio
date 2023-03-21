
# Virtual-Tryon介绍

图片描述任务：给定模特图、骨架图、衣服平铺图生成模特试衣图

## 模型描述

*我们提出一种新的单阶段端到端的虚拟试衣框架，并提出一种新的空间变换结构，在虚拟试衣和一些变换任务上达到SOTA*

## 期望模型使用方式以及适用范围


### 如何使用

在ModelScope里可以比较方便的使用Virtual-tryon的能力。

#### 代码范例

```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_daflow_virtual-try-on_base'
input_imgs = {
      'masked_model': 'https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_model.jpg',
      'pose': 'https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_pose.jpg',
      'cloth': 'https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_cloth.jpg'
}

pipeline_virtual_tryon = pipeline(
task=Tasks.virtual_try_on, model=model_id)
img = pipeline_virtual_tryon(input_imgs)[OutputKeys.OUTPUT_IMG]
cv2.imwrite('demo.jpg', img[:, :, ::-1])
```


### 模型局限性以及可能的偏差

模型在数据集上训练，在与训练数据差异大的数据测试效果会差，需要重新训练

## 训练数据介绍

数据会包含成对的模特图和对应模特衣服的

## 模型训练流程

暂时不支持通过ModelScope接口进行训练，敬请期待。

### 预处理

主要是用的预处理如下：

- 图像缩放到256*192分辨率

### 训练

暂不支持。

## 数据评估及结果

- 量化结果

![table](https://modelscope.cn/api/v1/models/damo/cv_daflow_virtual-try-on_base/repo?Revision=master&FilePath=./figs/table.jpg&View=true)

- VITON dataset

![viton](https://modelscope.cn/api/v1/models/damo/cv_daflow_virtual-try-on_base/repo?Revision=master&FilePath=./figs/fig1.jpg&View=true)

- MPV dataset

![mpv](https://modelscope.cn/api/v1/models/damo/cv_daflow_virtual-try-on_base/repo?Revision=master&FilePath=./figs/mpv.jpg&View=true)

- FashionVideo

![fashion](./figs/fashion.gif)

- ShapeNet

![shapenet](https://modelscope.cn/api/v1/models/damo/cv_daflow_virtual-try-on_base/repo?Revision=master&FilePath=./figs/shapenet.jpg&View=true)



### 相关论文以及引用信息

如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```
@article{bai2022single,
title={Single Stage Virtual Try-on via Deformable Attention Flows},
author={Bai, Shuai and Zhou, Huiling and Li, Zhikang and Zhou, Chang and Yang, Hongxia},
journal={arXiv preprint arXiv:2207.09161},
year={2022}
}
```