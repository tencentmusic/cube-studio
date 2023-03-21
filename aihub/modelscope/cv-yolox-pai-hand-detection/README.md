
# 手部检测模型
输入一张图像，并对其中手部区域进行检测，输出所有手部区域检测框、置信度和标签。

## 模型描述
该模型主要用于手部检测任务，从图像中检测出人手框坐标、置信度和标签。该任务使用阿里云PAI-EasyCV框架下的YOLOX-PAI模型在TV-hand和coco-hand-big综合数据集上训练而来，YOLOX-PAI从Backbone（repvgg backbone）、Neck（ gsconv/asff）、Head（toods/rtoods）、Loss（siou/giou）4个方向对原版的YOLOX进行优化，结合阿里巴巴计算平台PAI自研的PAI-Blade推理加速框架优化模型性能，在速度和精度上都比现阶段的40~50mAP的SOTA的YOLOv6更胜一筹。关于YOLOX-PAI细节请参考https://github.com/alibaba/EasyCV/blob/master/docs/source/tutorials/yolox.md。

yolox-pai论文参考https://arxiv.org/abs/2208.13040

## 使用方式和范围
使用方式：
- 输入任意图像，返回图像中所有的人手框坐标、置信度和标签。

目标场景:
1. 手势关键点。
2. 手势识别。
3. 手部重建。
4. 手势自然交互。

### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用来完成手部关键点检测任务。

#### 推理代码范例
也可以参考示例代码tests/pipelines/test_hand_detection.py

```python
# numpy >= 1.20
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_yolox-pai_hand-detection'
hand_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)
output = hand_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/hand_detection.jpg')

# the output contains boxes, scores and labels
print(output)
```

#### 微调代码范例
也可以参考示例代码tests/trainers/easycv/test_easycv_trainer_hand_detection.py

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
from modelscope.utils.constant import DownloadMode, LogKeys, Tasks
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

model_id = 'damo/cv_yolox-pai_hand-detection'
cfg_options = {'train.max_epochs': 2}

temp_file_dir = tempfile.TemporaryDirectory()
tmp_dir = temp_file_dir.name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

trainer_name = Trainers.easycv
train_dataset = MsDataset.load(
    dataset_name='hand_detection_dataset',
    split='subtrain')
eval_dataset = MsDataset.load(
    dataset_name='hand_detection_dataset',
    split='subtrain')
kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir,
    cfg_options=cfg_options)
trainer = build_trainer(trainer_name, kwargs)
trainer.train()

# model save path: results_files
results_files = os.listdir(tmp_dir)
# train log: json_files
json_files = glob.glob(os.path.join(tmp_dir, '*.log.json'))
temp_file_dir.cleanup()
```

### 模型局限性以及可能的偏差

- 输入图像存在人手严重残缺或遮挡的情形下，模型会出现误检的现象。
- 高速运动模糊的情形下，模型会出现人手误检的现象。


## 训练数据介绍
训练数据来自公开数据集COCO-HAND_Big和TV_HAND，作者已经整理好并转换成coco格式，地址是https://www.modelscope.cn/datasets/modelscope/hand_detection_dataset/summary

## 数据评估及结果
### 测评指标
模型在公开测试数据集上的评价指标、模型大小、参数量如下：

| 输入大小 | AR@1 | AR@10 | AR@100 |  AR@100 (small) | AR@100(medium) | AR@100(large) |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | 
| 640x640x3 | **0.2454** | **0.4295** | **0.4334** | **0.3884** | **0.5154** | **0.4978** |

| 输入大小 | mAP | mAP@.50IOU | mAP@.75IOU |  mAP (small) | mAP (medium) | mAP(large) |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | 
| 640x640x3 | **0.3526** | **0.7294** | **0.3035** | **0.3002** | **0.4414** | **0.4218** |

### 模型效果
![手部检测结果](https://modelscope.cn/api/v1/models/damo/cv_yolox-pai_hand-detection/repo?Revision=master&FilePath=assets/007888_006.jpg&View=true)
![手部检测结果](https://modelscope.cn/api/v1/models/damo/cv_yolox-pai_hand-detection/repo?Revision=master&FilePath=assets/047062_003.jpg&View=true)
![手部检测结果](https://modelscope.cn/api/v1/models/damo/cv_yolox-pai_hand-detection/repo?Revision=master&FilePath=assets/000000184320.jpg&View=true)

### 引用
```BibTeX
@article{DBLP:journals/corr/abs-2107-08430,
  title     = {YOLOX: Exceeding YOLO Series in 2021},
  author    = {Zheng Ge and Songtao Liu and Feng Wang and Zeming Li and Jian Sun},
  journal   = {arXiv preprint arXiv:2107.08430},
  year      = {2021}
}

@article{DBLP:journals/corr/abs-2208-13040,
  title     = {YOLOX-PAI: An Improved YOLOX Version by PAI[J]},
  author    = {Zou X, Wu Z, Zhou W, et al.},
  journal   = {arXiv preprint arXiv:2208.13040},
  year      = {2022}
}
```

#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/cv_yolox-pai_hand-detection.git
```
