

# Segformer-B0语义分割模型介绍

**其它相关模型体验**[Mask2Former-R50全景分割](https://www.modelscope.cn/models/damo/cv_r50_panoptic-segmentation_cocopan/summary) 

## 模型描述
Neurips2021文章SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers在COCO_Stuff164K数据集上的复现。官方源码暂没有提供COCO_Stuff164K的相关实现

本模型基于Segformer分割框架所配置训练的实时语义分割框架。使用了一个GPU上实时运行的配置结构。在CoCo-Stuff-164的数据集上进行了172类的分类

结构图如下：

<p align="left">
    <img src="description/framework.png" alt="donuts" />
  
本模型一共包含B0-B5一共6个不同模型，模型配置细节如下：

<p align="left">
    <img src="description/details.png" alt="donuts" />

B0-B5模型所在链接如下：

| 模型  |  链接                           | 
| ---------- |  ------------------------ | 
| SegFormer_B0 | https://www.modelscope.cn/models/damo/cv_segformer-b0_image_semantic-segmentation_coco-stuff164k/summary |
| SegFormer_B1 | https://www.modelscope.cn/models/damo/cv_segformer-b1_image_semantic-segmentation_coco-stuff164k/summary |
| SegFormer_B2 | https://www.modelscope.cn/models/damo/cv_segformer-b2_image_semantic-segmentation_coco-stuff164k/summary |
| SegFormer_B3 | https://www.modelscope.cn/models/damo/cv_segformer-b3_image_semantic-segmentation_coco-stuff164k/summary |
| SegFormer_B4 | https://www.modelscope.cn/models/damo/cv_segformer-b4_image_semantic-segmentation_coco-stuff164k/summary |
| SegFormer_B5 | https://www.modelscope.cn/models/damo/cv_segformer-b5_image_semantic-segmentation_coco-stuff164k/summary |


## 期望模型使用方式以及适用范围

本模型适用范围较广，能对图片中包含的大部分类别COCO 172类）进行语义分割。

### 如何使用
需要numpy版本大于1.20:  pip install numpy >=1.20

在ModelScope框架上，提供输入图片，既可通过简单的Pipeline调用来使用。



#### 代码范例

预测代码

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_semantic_segmentation.jpg'
segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_segformer-b0_image_semantic-segmentation_coco-stuff164k')
result = segmentation_pipeline(img)

print(f'segmentation output: {result}.')
```

训练代码 （训练时间很长）
```python
import tempfile
import os

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

model = 'damo/cv_segformer-b0_image_semantic-segmentation_coco-stuff164k'
#注意：本代码和配置仅仅作为示例使用。如果需要复现结果，请结合模型文件中的配置文件，使用8卡并行方式复现

trainer_name = Trainers.easycv
train_dataset = MsDataset.load(
    dataset_name='coco_stuff164k',
    namespace='damo',
    split='train') 
eval_dataset = MsDataset.load(
    dataset_name='coco_stuff164k',
    namespace='damo',
    split='validation') 
kwargs = dict(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir,
    )

trainer = build_trainer(trainer_name, kwargs)
trainer.train()
```


### 模型局限性以及可能的偏差

- 部分非常规图片或感兴趣物体占比太小或遮挡严重可能会影响分割结果
- 对新类别（集外类别）不兼容

## 训练数据介绍

- [COCO-Stuff 164K](https://github.com/nightrome/cocostuff)： COCO-Stuff augments the popular COCO dataset with pixel-level stuff annotations. These annotations can be used for scene understanding tasks like semantic segmentation, object detection and image captioning.


## 模型训练流程

- 使用论文推荐的训练方式，在COCO上使用AdamW优化器，初始学习率为6e-5；

### 预处理

测试时主要的预处理如下：
- Resize：先将原始图片的短边Resize到512，等比例缩放。此时如果长边超过了2048，则按照最长边为2048，重新计算Resize的scale进行Resize
- Normalize：图像归一化，减均值除以标准差
- Pad：图像高宽补零至32的倍数

## 数据评估及结果

## SegFormer

Semantic segmentation models trained on **CoCo_stuff164k** val集合上的性能如下.

| Algorithm  |  Params<br/>(backbone/total)                            | inference time(V100)<br/>(ms/img)                    |mIoU |
| ---------- |  ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SegFormer_B0 | 3.3M/3.8M | 47.2ms |  35.91               |


## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@article{xie2021segformer,
  title={SegFormer: Simple and efficient design for semantic segmentation with transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={12077--12090},
  year={2021}
}
```


