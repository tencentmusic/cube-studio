
# 日常物体识别模型介绍
自建1300类常见物体标签体系，覆盖常见的日用品，动物，植物，家具，设备，食物等物体，标签从海量中文互联网社区语料进行提取，保留了出现频率较高的常见物体名称。模型结构采用最新的ViT-Base结构。  
创空间快速可视化展示: [ViT图像分类-中文-日常物品](https://modelscope.cn/studios/tany0699/cv_vit-base_image-classification_Dailylife-labels/summary)  
  
本系列还有如下模型，欢迎试用:  
- [ViT图像分类-通用](https://modelscope.cn/models/damo/cv_vit-base_image-classification_ImageNet-labels/summary)
- [NextViT实时图像分类-中文-日常物品](https://modelscope.cn/models/damo/cv_nextvit-small_image-classification_Dailylife-labels/summary)
- [ConvNeXt图像分类-中文-垃圾分类](https://modelscope.cn/models/damo/cv_convnext-base_image-classification_garbage/summary)
- [BEiTv2图像分类-通用-base](https://modelscope.cn/models/damo/cv_beitv2-base_image-classification_patch16_224_pt1k_ft22k_in1k/summary)
- [BEiTv2图像分类-通用-large](https://modelscope.cn/models/damo/cv_beitv2-large_image-classification_patch16_224_pt1k_ft22k_in1k/summary)

## 模型描述

采用Transformer经典的[ViT-Base](https://github.com/google-research/vision_transformer)结构, 并采用了DeiT的知识蒸馏方式进行训练。  
<img src="./resources/overview.jpg" alt="overview"/>

## 期望模型使用方式以及适用范围

本模型适用范围较广，覆盖大部分日常生活常见的物品类目，包括日用品，动物，植物，家具，设备，食物等。也可作为下游任务的预训练backbone。

### 如何使用

在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/bird.JPEG'
image_classification = pipeline(Tasks.image_classification, 
                                model='damo/cv_vit-base_image-classification_Dailylife-labels')
result = image_classification(img_path)
print(result)
```

### 模型局限性以及可能的偏差

- 支持1300类常见物体识别


## 训练数据介绍

- 140万包含常见物体的图像集


## 模型训练流程

- 主要训练参数参考[DeiT论文](https://arxiv.org/abs/2012.12877)的设置，除了weight decay在复现时设置为0.1，模型训练未使用pretrained参数进行初始化。

### 预处理

测试时主要的预处理如下：
- Resize：先将原始图片的短边缩放至256
- Normalize：图像归一化，减均值除以标准差
- CenterCrop：裁切为224x224

## 数据评估及结果

模型在自建测试集进行测试，结果如下:

| Model | top-1 acc | top-5 acc | #params  | Remark       | 
|:--------:|:-------:|:--------:|:-------:|--------------|
|  ViT-base  | 74.3   |  95.3   |  86M   | modelscope |


## 模型训练
使用托管在modelscope DatasetHub上的小型数据集[mini_imagenet100](https://modelscope.cn/datasets/tany0699/mini_imagenet100/summary)进行finetune训练的示例代码: 

```python
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
import tempfile

model_id = 'damo/cv_vit-base_image-classification_Dailylife-labels'

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
    cfg.train.dataloader.batch_size_per_gpu = 32 # batch大小
    cfg.train.dataloader.workers_per_gpu = 2     # 每个gpu的worker数目
    cfg.train.max_epochs = 1                     # 最大训练epoch数
    cfg.model.mm_model.head.num_classes = 100                     # 分类数
    cfg.model.mm_model.train_cfg.augments[0].num_classes = 100    # 分类数
    cfg.model.mm_model.train_cfg.augments[1].num_classes = 100    # 分类数
    cfg.train.optimizer.lr = 1e-4                # 学习率
    cfg.train.lr_config.warmup_iters = 1         # 预热次数
    cfg.train.evaluation.metric_options = {'topk': (1, 5)}  # 训练时的评估指标
    cfg.evaluation.metric_options = {'topk': (1, 5)}        # 评估时的评估指标
    return cfg

# 构建训练器
kwargs = dict(
    model=model_id,                 # 模型id
    work_dir=tmp_dir,               # 工作目录
    train_dataset=ms_train_dataset, # 训练集  
    eval_dataset=ms_val_dataset,    # 验证集
    cfg_modify_fn=cfg_modify_fn,    # 用于修改训练配置文件的回调函数
    model_revision='v1.0.2'
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

model_id = 'damo/cv_vit-base_image-classification_Dailylife-labels'

# 加载用于评估的数据集
ms_val_dataset = MsDataset.load(
            'dailytags', namespace='tany0699',
            subset_name='default', split='validation') 

tmp_dir = tempfile.TemporaryDirectory().name # 使用临时目录作为工作目录

# 构建训练器
kwargs = dict(
    model=model_id,                 # 模型id
    work_dir=tmp_dir,               # 工作目录
    train_dataset=None,  
    eval_dataset=ms_val_dataset,    # 评估的数据集
    model_revision='v1.0.2'
    )
trainer = build_trainer(name=Trainers.image_classification, default_args=kwargs)

# 开始评估
result = trainer.evaluate()
print('result:', result)
```
评估过程默认使用模型中自带的预训练权重进行评估。  

#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/cv_vit-base_image-classification_Dailylife-labels.git
```