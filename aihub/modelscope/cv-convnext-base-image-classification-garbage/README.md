
# 生活垃圾分类模型介
自建265类常见的生活垃圾标签体系，15w张图片数据，包含可回收垃圾、厨余垃圾、有害垃圾、其他垃圾4个标准垃圾大类，覆盖常见的食品，厨房用品，家具，家电等生活垃圾小类共265个，标签从海量中文互联网社区语料进行提取，整理出频率较高的常见生活垃圾名称。模型结构采用ConvNeXt-Base结构, 先在大规模数据集ImageNet-22K上预训练，后在数据集上进行微调，最终垃圾分类验证集上top-1精度为92.23%。  
垃圾分类:  
<img src="./resources/trash.jpg" width = "640" height = "512" alt="垃圾分类"/>  
  
本系列还有如下模型，欢迎试用:  
- [ViT图像分类-中文-日常物品](https://modelscope.cn/models/damo/cv_vit-base_image-classification_Dailylife-labels/summary)
- [ViT图像分类-通用](https://modelscope.cn/models/damo/cv_vit-base_image-classification_ImageNet-labels/summary)
- [NextViT实时图像分类-中文-日常物品](https://modelscope.cn/models/damo/cv_nextvit-small_image-classification_Dailylife-labels/summary)
- [BEiTv2图像分类-通用-base](https://modelscope.cn/models/damo/cv_beitv2-base_image-classification_patch16_224_pt1k_ft22k_in1k/summary)
- [BEiTv2图像分类-通用-large](https://modelscope.cn/models/damo/cv_beitv2-large_image-classification_patch16_224_pt1k_ft22k_in1k/summary)

## 模型描述
模型结构采用[ConvNeXt-Base](https://arxiv.org/abs/2201.03545)，是一个全面超越Swin Transformer的CNN。论文从原始的ResNet出发，通过借鉴Swin Transformer的设计来逐步地改进ResNet模型，并测试了纯卷积网络所能达到的极限，在这个过程中发现了几个有助于性能提高的关键组件。ConvNeXt完全由标准ConvNet模块构建，但在准确性和可扩展性方面与transformer相比具有竞争性，它实现了87.8%的ImageNet top-1精度，并在COCO检测和ADE20K分割任务上优于Swin transformer，同时保持了标准ConvNet的简单性和高效性。  
[论文](https://arxiv.org/abs/2201.03545) | [代码](https://github.com/facebookresearch/ConvNeXt)


## 期望模型使用方式以及适用范围
本模型适用于日常生活垃圾分类，可识别可回收垃圾、厨余垃圾、有害垃圾、其他垃圾这4个标准的垃圾大类，265个垃圾小类，覆盖常见的食品，厨房用品，家具，家电等生活垃圾。也可作为下游任务的预训练backbone。

### 如何使用
在ModelScope框架上，提供输入图片，即可通过简单的Pipeline调用来使用。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/banana.jpg'
image_classification = pipeline(Tasks.image_classification, 
                                model='damo/cv_convnext-base_image-classification_garbage')
result = image_classification(img_path)
print(result)
```

### 模型局限性以及可能的偏差
- 支持识别4个标准的生活垃圾大类：可回收垃圾、厨余垃圾、有害垃圾、其他垃圾
- 支持识别265个生活垃圾品类


## 训练数据介绍
- 15万包含常见的生活垃圾图像集，包含可回收垃圾、厨余垃圾、有害垃圾、其他垃圾4个标准垃圾大类，覆盖常见的食品，厨房用品，家具，家电等265个垃圾小类。其中训练集133038张图像，验证集14642张图像，均从海量中文互联网社区语料进行提取，数据大小为13GB。


## 模型训练流程
- 主要训练参数参考论文[ConvNeXt](https://arxiv.org/abs/2201.03545)的设置，模型先在大规模数据集ImageNet-22K上预训练后，再在垃圾数据集上进行微调，微调使用4张GPU，batchsize为128，epoch为100，lr为5e-5，weight_decay=0.1, warmup_iters=10, 其它参数不变。

### 预处理
测试时主要的预处理如下：
- Resize：先将原始图片的短边缩放至256
- Normalize：图像归一化，减均值除以标准差
- CenterCrop：裁切为224x224

## 数据评估及结果

模型在自建测试集进行测试，结果如下:

| Model | top-1 acc | top-5 acc | #params  | Remark       | 
|:--------:|:-------:|:--------:|:-------:|--------------|
|  ConvNeXt-Base  | 92.23   |  98.20   |  88M   | modelscope |


## 模型训练
使用托管在modelscope DatasetHub上的小型数据集[mini_imagenet100](https://modelscope.cn/datasets/tany0699/mini_imagenet100/summary)进行finetune训练的示例代码: 
```python
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
import tempfile

model_id = 'damo/cv_convnext-base_image-classification_garbage'

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
    cfg.model.mm_model.head.num_classes = 100    # 分类数
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
对[中文生活垃圾分类数据集](https://modelscope.cn/datasets/tany0699/garbage265/summary)进行精度评估示例代码如下:
```python
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
import tempfile

model_id = 'damo/cv_convnext-base_image-classification_garbage'

# 加载用于评估的数据集
ms_val_dataset = MsDataset.load(
            'garbage265', namespace='tany0699',
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
评估过程默认使用模型中自带的预训练权重进行评估, 评估结果为: result: {'accuracy_top-1': 92.23241424560547, 'accuracy_top-5': 98.20011138916016}

## Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/cv_convnext-base_image-classification_garbage.git
```


## 引用
如果该模型对你有所帮助，请引用相关的论文：
```
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
```