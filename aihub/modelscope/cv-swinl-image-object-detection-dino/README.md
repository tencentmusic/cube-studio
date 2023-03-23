
<!--- 以下model card模型说明部分，请使用中文提供（除了代码，bibtex等部分） --->

# DINO-高精度目标检测模型介绍
本模型为基于DINO算法的高精度目标检测模型。


## 模型描述

DINO模型算法框架图如下，是改进的DETR系列模型。

<img src="https://modelscope.cn/api/v1/models/damo/cv_swinl_image-object-detection_dino/repo?Revision=master&FilePath=assets/dino_framework.jpg&View=true" width="800" >


## 期望模型使用方式以及适用范围
该模型适用于通用图像目标检测，输入图像，输出图像中检测到的物体位置信息。
该模型暂不支持CPU推理和训练，需要在GPU机器上运行。

### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型，得到图像中物体的矩形框信息。

#### 代码范例
基础示例代码。下面的示例代码展示的是如何通过一张图片作为输入，得到图片对应的吸烟检测结果。
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_swinl_image-object-detection_dino'
test_image = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_detection.jpg'
image_object_detection_dino = pipeline(Tasks.image_object_detection, model=model_id)
result = image_object_detection_dino(test_image)
print("result is : ", result)
```

## 训练数据介绍
本模型是基于开源数据集COCO训练得到。

## 数据评估及结果
模型在COCO的验证集上客观指标如下：

| Method | bbox_mAP| AP@0.5 | inference time(V100)| Parameters (backbone/total)|
| ------------ | ------------ | ------------ | ------------ | ------------ |
| DINO | 63.39 | 80.25 | 325ms | 195M/218M  |

## 模型训练流程
可在用户的目标数据集上进行finetune, 以人头数据集为例，在GPU机器上finetune代码如下：

```python
# 模块引入
import os
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from modelscope.hub.snapshot_download import snapshot_download

# 数据集准备
train_dataset = MsDataset.load(
    'head_detection_for_train_tutorial', 
    namespace="modelscope", 
    split='train', 
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
val_dataset = MsDataset.load(
    'head_detection_for_train_tutorial', 
    namespace="modelscope", 
    split='validation', 
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

# 相关参数设置
model_id = 'damo/cv_swinl_image-object-detection_dino'
work_dir = "./output/cv_swinl_image-object-detection_dino"
trainer_name = Trainers.easycv
total_epochs = 15
class_names = ['head']
cfg_options = {
    'train.max_epochs':
    total_epochs,
    'train.hooks': [
        {
            'type': 'CheckpointHook',
            'interval': 15
        },
        {
            'type': 'EvaluationHook',
            'interval': 15
        },
        {
            'type': 'TextLoggerHook',
            'ignore_rounding_keys': None,
            'interval': 1
        },
    ],
    'dataset.train.data_source.classes': class_names,
    'dataset.val.data_source.classes': class_names,
    'evaluation.metrics.evaluators': [
        {
            'type': 'CocoDetectionEvaluator',
            'classes': class_names
        }
    ],
    'CLASSES': class_names,
}

# 新建trainer
kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    work_dir=work_dir,
    cfg_options=cfg_options)

print("build trainer.")
trainer = build_trainer(trainer_name, kwargs)
print("start training.")

# 预训练模型下载
cache_path = snapshot_download(model_id)

# 开启训练
trainer.train(
    checkpoint_path=os.path.join(cache_path, 'pytorch_model.pt'),
    load_all_state=False,
)

# 模型评估
eval_res = trainer.evaluate(checkpoint_path=os.path.join(work_dir, 'epoch_15.pth'))
print(eval_res)
```

### 模型局限性以及可能的偏差
- 该模型暂不支持CPU推理和训练，需要在GPU机器上运行。

### 相关论文以及引用信息
本模型主要参考论文如下（论文链接：[link](https://arxiv.org/abs/2203.03605)）：

```BibTeX
 @article{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
