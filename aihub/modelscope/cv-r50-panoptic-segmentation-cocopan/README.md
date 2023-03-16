
# Mask2Former r50-image-panoptic-segmentation模型介绍
给定一张输入图像，输出全景分割掩膜，类别，分数（虚拟分数）。

全景分割是要分割出图像中的stuff，things。stuff是指天空，草地等不规则区域，things是指可数的物体，例如人，车，猫等。

Mask2Former是一种能够解决任何图像分割任务（全景、实例或语义）的新架构。它包含了一个masked attention结构，通过将交叉注意力计算内来提取局部特征。

![Mask2Former模型结构](description/mask2former.png)

## 模型描述
本模型使用resnet50为backbone，Mask2Former为分割头。COCO全景分割数据库上训练。

resnet50的backbone占用显存比swin-large小很多，能够使用16G显存的GPU进行训练。

## 期望模型使用方式与适用范围
本模型适用范围较广，能对图片中包含的大部分感兴趣物体（COCO things 80类，stuff 53类）进行分割。
### 如何使用
在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。
#### 代码范例
**由于模型中使用MultiScaleDeformableAttention，暂不支持CPU，请使用GPU的实例运行**

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

segmentor = pipeline(Tasks.image_segmentation, model='damo/cv_r50_panoptic-segmentation_cocopan')
input_url = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_panoptic_segmentation.jpg'
result = segmentor(input_url)
```

```shell
PYTHONPATH=. python tests/run.py --pattern test_panoptic_mask2former.py
```


## 模型训练流程
```python
import glob
import os
import shutil
import tempfile
import unittest

import torch

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import LogKeys, Tasks
from modelscope.utils.test_utils import test_level

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

cfg_options = {'train.max_epochs': 2}

trainer_name = Trainers.easycv
train_dataset = MsDataset.load(dataset_name='COCO2017_panopic_subset', namespace='modelscope', split='train')
eval_dataset = MsDataset.load(dataset_name='COCO2017_panopic_subset', namespace='modelscope', split='validation')
kwargs = dict(
    model='damo/cv_r50_panoptic-segmentation_cocopan',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir,
    cfg_options=cfg_options)

trainer = build_trainer(trainer_name, kwargs)
trainer.train()
```

也可通过unitest代码直接调取

```shell
PYTHONPATH=. python tests/run.py --pattern test_easycv_trainer_panoptic_mask2former.py
```

### 模型局限性以及可能的偏差
- 部分非常规图片或感兴趣物体占比太小或遮挡严重可能会影响分割结果
- 当前版本在python 3.7环境测试通过，其他环境下可用性待测试
## 训练数据介绍
- [COCO-panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) ：COCO全称是Common Objects in Context，是Microsoft团队提供的一个可以用来图像识别、检测和分割的数据集。COCO-panoptic 2017 包含前景（things）80个类别，背景（stuff）53个类别。
### 预处理
测试时主要的预处理如下：
- Resize：先将原始图片的短边Resize到800，等比例缩放。此时如果长边超过了1333，则按照最长边为1333，重新计算Resize的scale进行Resize
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数

## 数据评估及结果
| Backbone |  Pretrain   | box mAP | mask mAP | PQ | 
|:--------:|:-----------:|:-------:|:--------:|:-------:|
| resnet50 | ImageNet-21K|  44.81   |   41.88   |  51.64   | 

## 引用
```BibTeX
@inproceedings{cheng2022masked,
  title={Masked-attention mask transformer for universal image segmentation},
  author={Cheng, Bowen and Misra, Ishan and Schwing, Alexander G and Kirillov, Alexander and Girdhar, Rohit},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1290--1299},
  year={2022}
}
```