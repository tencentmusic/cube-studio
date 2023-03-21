

# 基础视觉模型高效调优：LoRA

基于大规模预训练基础模型的参数高效迁移学习方法在各种下游应用中均取得了优异的表现，其中包括了利用LoRA进行调优的方法。该方法对多头注意力层中的映射权重额外添加了低秩矩阵的训练模块，仅需训练极少部分的参数，就能取得不错的性能表现。

该页面展示了LoRA在图像分类任务上的应用，即给定一张图片，返回候选类别中的分类标签及置信度。


## 模型描述
Prompt的模型结构如下所示，其中左侧为LoRA嵌入到Vision Transformer中的框架，右侧为LoRA的具体结构：

<img src="./description/lora_architecture.png" alt="architecture" width="40%" height="40%">

## 期望模型使用方式以及适用范围

### 如何使用

基于 ModelScope 框架，通过调用预定义的 Pipeline 可实现快速调用。


#### 代码范例

```python
from modelscope.pipelines import pipeline

lora_pipeline = pipeline('vision-efficient-tuning',
                         'damo/cv_vitb16_classification_vision-efficient-tuning-lora',
                          model_revision='v1.0.2')
result = lora_pipeline('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/vision_efficient_tuning_test_1.png')
print(f'Output: {result}.')            
```


### 模型局限性以及可能的偏差

- 本模型基于公开的[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)通用数据集训练，且仅适用于训练数据的覆盖类别，在具体应用场景下可能存在偏差。
- 本模型当前仅用于图像分类任务，同时该方法可用于其他模态输入（如文本、视频等）和其他视觉下游任务（如检测、分割等）。

## 训练数据介绍

1. [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) 通用图像分类数据集，包含100个类别。
2. [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011) 鸟类细粒度分类数据集，包含200个类别。
3. [NABirds](https://dl.allaboutbirds.org/nabirds) 鸟类细粒度分类数据集，包含555个类别。
4. [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) 花卉细粒度分类数据集，包含102个类别。
5. [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) 车辆细粒度分类数据集，包含196个类别。
6. [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) 犬类细粒度分类数据集，包含120个类别。


## 数据评估及结果

模型分别在不同的预训练模型和图像分类数据集下进行评估，结果如下：

|     Dataset     | ViT-B/16 (IN-21K) | ViT-L/14 (CLIP) |
|:---------------:|:-----------------:|:---------------:|
|     CIFAR100    |       92.24%      |      90.96%     |
|   CUB-200-2011  |       88.09%      |      87.69%     |
|     NABirds     |       84.11%      |      86.57%     |
| Oxford Flowers  |       98.94%      |      95.45%     |
|  Stanford Cars  |       85.82%      |      92.30%     |
|  Stanford Dogs  |       89.81%      |      88.31%     |
|     Average     |       89.84%      |      90.21%     |

其中，ViT-B/16模型使用 [ImageNet-21K](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) 作为预训练模型，ViT-L/14使用 [CLIP](https://github.com/openai/CLIP) 作为预训练模型。

## 模型训练和验证

以下为使用[FME Benchmark](https://modelscope.cn/datasets/damo/foundation_model_evaluation_benchmark/summary)中的子数据集[OxfordFlowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)[[点击预览](https://modelscope.cn/datasets/damo/foundation_model_evaluation_benchmark/dataPeview)]进行finetune训练和评测的示例代码：

```python
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode

# 模型ID
model_id = 'damo/cv_vitb16_classification_vision-efficient-tuning-lora'

# 加载训练集
ms_train_dataset = MsDataset.load(
    'foundation_model_evaluation_benchmark', 
    namespace='damo',
    subset_name='OxfordFlowers', 
    split='train',
	download_mode=DownloadMode.FORCE_REDOWNLOAD)   

# 加载验证集
ms_eval_dataset = MsDataset.load(
    'foundation_model_evaluation_benchmark', 
    namespace='damo',
    subset_name='OxfordFlowers', 
    split='eval',
	download_mode=DownloadMode.FORCE_REDOWNLOAD)      

tmp_dir = tempfile.TemporaryDirectory().name # 使用临时目录作为工作目录

# 修改配置文件
def cfg_modify_fn(cfg):
    max_epochs = 1                            # 最大训练轮次
    cfg.model.head.num_classes = 102          # 类别数
    cfg.model.finetune = True                 # 进行微调
    cfg.train.max_epochs = max_epochs         # 最大训练轮次
    cfg.train.lr_scheduler.T_max = max_epochs # 学习率调度器的参数
    cfg.model.backbone.lora_length = 10       # 模型超参数
    return cfg

# 构建训练器
kwargs = dict(
    model=model_id,                 # 模型id
    work_dir=tmp_dir,               # 工作目录
    train_dataset=ms_train_dataset, # 训练集  
    eval_dataset=ms_eval_dataset,   # 验证集
    cfg_modify_fn=cfg_modify_fn     # 用于修改训练配置文件的回调函数
)
trainer = build_trainer(name=Trainers.vision_efficient_tuning, default_args=kwargs)

# 进行训练
trainer.train()

# 进行评估
result = trainer.evaluate()
print('result:', result)
```

训练说明见示例代码中的注释部分，详细的训练说明和用法见官方的[训练文档](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train)。

## 相关论文以及引用信息


如果该模型对您有所帮助，请引用下面的相关的论文：

```BibTeX
@inproceedings{hu2021lora,
  title={{LoRA}: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward and Shen, Yelong and Wallis, Phil and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Lu and Chen, Weizhu},
  booktitle=ICLR,
  year={2021}
}
```