
# 通用图像分类模型介绍
DeiT-base复现，采用ImageNet数据训练。  
创空间快速可视化展示: [ViT图像分类-通用](https://modelscope.cn/studios/tany0699/cv_vit-base_image-classification_ImageNet-labels/summary)   
  
本系列还有如下模型，欢迎试用:  
- [ViT图像分类-中文-日常物品](https://modelscope.cn/models/damo/cv_vit-base_image-classification_Dailylife-labels/summary)
- [NextViT实时图像分类-中文-日常物品](https://modelscope.cn/models/damo/cv_nextvit-small_image-classification_Dailylife-labels/summary)
- [ConvNeXt图像分类-中文-垃圾分类](https://modelscope.cn/models/damo/cv_convnext-base_image-classification_garbage/summary)
- [BEiTv2图像分类-通用-base](https://modelscope.cn/models/damo/cv_beitv2-base_image-classification_patch16_224_pt1k_ft22k_in1k/summary)
- [BEiTv2图像分类-通用-large](https://modelscope.cn/models/damo/cv_beitv2-large_image-classification_patch16_224_pt1k_ft22k_in1k/summary)

## 模型描述

采用Transformer经典的[ViT-Base](https://github.com/google-research/vision_transformer)结构

## 期望模型使用方式以及适用范围

本模型适用范围较广，支持ImageNet 1000类物体识别，也可作为下游任务的预训练backbone

### 如何使用

在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/bird.JPEG'
image_classification = pipeline(Tasks.image_classification,
                                model='damo/cv_vit-base_image-classification_ImageNet-labels')
result = image_classification(img_path)
print(result)
```

### 模型局限性以及可能的偏差

- 只支持ImageNet-1K标签覆盖到的物体识别


## 训练数据介绍

- [ImageNet-1k](https://ieeexplore.ieee.org/document/5206848)：ImageNet数据集包含14,197,122个带注释的图像。自2010年以来，作为图像分类的基准数据集，该数据集被用于ImageNet大规模视觉识别挑战(ILSVRC)。


## 模型训练流程

- 主要训练参数遵循[DeiT论文](https://arxiv.org/abs/2012.12877)的设置，除了weight decay在复现时设置为0.1，Top-1精度比论文结果提升0.4

### 预处理

测试时主要的预处理如下：
- Resize：先将原始图片的短边缩放至256
- Normalize：图像归一化，减均值除以标准差
- CenterCrop：裁切为224x224

## 数据评估及结果

模型在ImageNet-1k val上进行测试，结果如下:

| Model | top-1 acc | top-5 acc | #params  | Remark       | 
|:--------:|:-------:|:--------:|:-------:|--------------|
|  DeiT-base  | 81.8   |  95.6   |  86M   | [official](https://github.com/facebookresearch/deit/blob/main/README_deit.md) |
|  DeiT-base  | 82.2   |   95.9   |  86M   |  modelscope   |

## 模型训练
使用托管在modelscope DatasetHub上的小型数据集[mini_imagenet100](https://modelscope.cn/datasets/tany0699/mini_imagenet100/summary)进行finetune训练的示例代码: 

```python
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
import tempfile

model_id = 'damo/cv_vit-base_image-classification_ImageNet-labels'

# 加载数据
ms_train_dataset = MsDataset.load(
            'mini_imagenet100', namespace='tany0699',
            subset_name='default', split='train')      # 加载训练集

ms_val_dataset = MsDataset.load(
            'mini_imagenet100', namespace='tany0699',
            subset_name='default', split='validation') # 加载验证集

tmp_dir = tempfile.TemporaryDirectory().name # 使用临时目录作为工作目录

# 修改配置文件
def cfg_modify_fn(cfg):
    cfg.train.dataloader.batch_size_per_gpu = 16 # batch大小
    cfg.train.dataloader.workers_per_gpu = 1     # 每个gpu的worker数目
    cfg.train.max_epochs = 1                     # 最大训练epoch数
    cfg.model.mm_model.head.num_classes = 100                     # 分类数
    cfg.model.mm_model.train_cfg.augments[0].num_classes = 100    # 分类数
    cfg.model.mm_model.train_cfg.augments[1].num_classes = 100    # 分类数
    cfg.train.optimizer.lr = 1e-4                # 学习率
    cfg.train.lr_config.warmup_iters = 1         # 预热次数
    return cfg

# 构建训练器
kwargs = dict(
    model=model_id,                 # 模型id
    work_dir=tmp_dir,               # 工作目录
    train_dataset=ms_train_dataset, # 训练集  
    eval_dataset=ms_val_dataset,    # 验证集
    cfg_modify_fn=cfg_modify_fn     # 用于修改训练配置文件的回调函数
    )
trainer = build_trainer(name=Trainers.image_classification, default_args=kwargs)

# 进行训练
trainer.train()

# 进行评估
result = trainer.evaluate()
print('result:', result)
```
训练说明见示例代码中的注释，更详细的训练说明和用法见官方的[训练文档](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train)。训练过程产生的log和模型权重文件保存在work_dir工作目录中，并以前缀为'best_'的文件保存了在验证集上最优精度的权重。evaluate()默认使用精度最高的模型权重进行评估。


## 模型评估
使用训练好的模型对需要评估的数据集进行精度评估示例代码如下:

```python
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
import tempfile

model_id = 'damo/cv_vit-base_image-classification_ImageNet-labels'

# 加载用于评估的数据集
ms_val_dataset = MsDataset.load(
            'imagenet_val', namespace='tany0699',
            subset_name='default', split='validation') 

tmp_dir = tempfile.TemporaryDirectory().name # 使用临时目录作为工作目录

# 构建训练器
kwargs = dict(
    model=model_id,                 # 模型id
    work_dir=tmp_dir,               # 工作目录
    train_dataset=None,  
    eval_dataset=ms_val_dataset     # 评估的数据集
    )
trainer = build_trainer(name=Trainers.image_classification, default_args=kwargs)

# 开始评估
result = trainer.evaluate()
print('result:', result)
```
评估过程默认使用模型中自带的预训练权重进行评估。


## 引用
如果你觉得这个该模型有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@InProceedings{pmlr-v139-touvron21a,
  title =     {Training data-efficient image transformers &amp; distillation through attention},
  author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
  booktitle = {International Conference on Machine Learning},
  pages =     {10347--10357},
  year =      {2021},
  volume =    {139},
  month =     {July}
}
```