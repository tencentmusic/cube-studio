
# 106点人脸关键点检测模型

输入一张人脸图像，实现人脸关键点检测，输出图像中人脸的106点关键点坐标和人像姿态角度。

## 106点人脸关键点
![人脸关键点定义](assets/keypoints.jpg)

## 模型描述
该模型主要用于人脸关键点检测和对齐任务，从包含人脸的图片中检测出人脸框、人脸关键点坐标和人脸姿态角。主要借鉴MobileNetV1和MobileNetV2的思路(如下图)，MobileNetV1速度快，放在浅层用于提取特征图，MobileNetV2速度相对慢但是信息保存好，用于提取深层语义信息，模型参数量少速度快，能良好应用在移动端实时人脸关键点检测场景。

![](assets/1.png)

## 使用方式和范围
使用方式：
- 输入包含人脸的图片，返回图像中所有的人脸框坐标，人脸关键点坐标，维度（106，2），和人头姿态角度，维度（1，3），分别是pitch,roll,yaw，pipeline支持360度人脸朝向下的检测任务。

目标场景:
1. 美颜特效：可用于直播、长短视频人像美颜、美妆、卡通画、换脸和特效等场景。
2. 人脸识别：可用于人脸识别和比对场景。
3. 人脸3D重建：基于2D人脸关键点的3D人脸重建和虚拟形象。

### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用来完成人脸关键点检测任务。

#### 推理代码范例
也可以参考示例代码tests/pipelines/test_face_2d_keypoints.py

```python
# numpy >= 1.20
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_mobilenet_face-2d-keypoints_alignment'
face_2d_keypoints = pipeline(Tasks.face_2d_keypoints, model=model_id)
output = face_2d_keypoints('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/keypoints_detect/test_img_face_2d_keypoints.png')

# the output contains point and pose
print(output)
```

#### 微调代码范例
也可以参考示例代码tests/trainers/easycv/test_easycv_trainer_face_2d_keypoints.py

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

model_id = 'damo/cv_mobilenet_face-2d-keypoints_alignment'
cfg_options = {'train.max_epochs': 2}

temp_file_dir = tempfile.TemporaryDirectory()
tmp_dir = temp_file_dir.name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

trainer_name = Trainers.easycv
train_dataset = MsDataset.load(
    dataset_name='face_2d_keypoints_dataset',
    namespace='modelscope',
    split='train',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
eval_dataset = MsDataset.load(
    dataset_name='face_2d_keypoints_dataset',
    namespace='modelscope',
    split='train',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir,
    cfg_options=cfg_options)
trainer = build_trainer(trainer_name, kwargs)
trainer.train()

        
results_files = os.listdir(tmp_dir)
json_files = glob.glob(os.path.join(tmp_dir, '*.log.json'))
temp_file_dir.cleanup()
```

### 模型局限性以及可能的偏差

- 输入图像存在人脸严重残缺或遮挡的情形下，模型会出现点位误检的现象。
- 输入图像人脸存在较大旋转角度时，模型会出现点位误检的现象。
- 高度运动模糊的情形下，模型会出现人体或点位误检的现象。
- 模型训练时主要基于单帧图像，在视频数据上，建议进行帧间点位平滑后处理。


## 训练数据介绍
训练数据包含公开数据集（COCO，AI Challenger等）、自采人脸图像视频，互联网搜集人脸图像视频等并进行标注作为训练数据。


## 数据评估及结果
### 测评指标
模型在自研测试数据集上的评价指标、模型大小、参数量如下：

| 输入大小 | POINTS-ION-NME | POSE-ME | MFLOPS |  PARAMS |
| ------------ | ------------ | ------------ | ------------ | ------------ | 
| 96x96 | **0.0981** | **10.5242** | **7.456383** | **0.266427 M** |

### 模型效果
![人体关键检测结果](assets/result_002253.png)
![人体关键检测结果](assets/result_002258.png)

### 引用
```BibTeX
@article{howard2017mobilenets,
  title={Mobilenets: Efficient convolutional neural networks for mobile vision applications},
  author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  journal={arXiv preprint arXiv:1704.04861},
  year={2017}
}

@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```
