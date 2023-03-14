
# 通用超轻量级检测模型
通用超轻量级检测目标检测的子任务，本模型为高性能通用实时检测模型，提供快速、精确的目标检测能力。

See the world fast and accurately!


## 模型描述

<img src="res/demo.png" >


YOLOX为YOLO检测系列的最近增强版本。在实时通用检测模型中，YOLO系列模型获得显著的进步，大大地推动了社区的发展。YOLOX在原有YOLO系列的基础上，结合无锚点（anchor-free）设计，自动优化GT分配（SimOTA）策略，分类回归头解耦（Decoupling Head）等一系列前沿视觉检测技术，显著地提高了检测模型的准确度与泛化能力，将当前的目标检测水平推到了一个新的高度。本模型为YOLOX的小规模模型，基于公开数据集COCO训练，支持80类通用目标检测。


<img src="res/git_fig.png" width="1000" >


## 期望模型使用方式以及适用范围

- 日常通用检测场景目标定位于识别。
- 移动端、边缘段日常物体检测。
- 作为其他日常场景算法的前置算法，如人体关键点检测，行为识别等。


### 如何使用

在ModelScope框架上，可以通过ModelScope的pipeline进行调用.

Now, you can play the model with a few line codes!

#### 代码范例一
Pipeline 调用示例
<!--- 本session里的python代码段，将被ModelScope模型页面解析为快速开始范例--->
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

realtime_detector = pipeline(Tasks.image_object_detection, model='damo/cv_cspnet_image-object-detection_yolox_nano_coco')
result = realtime_detector('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/keypoints_detect/000000438862.jpg')
# bbox
print(result)
```

#### 代码范例二
使用数据集迭代调用Pipeline示例
```python
import os
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

realtime_detector = pipeline(Tasks.image_object_detection, model='damo/cv_cspnet_image-object-detection_yolox_nano_coco')

dataset = MsDataset.load("cv_realtime-image-object-detection_TestDataset", namespace="damo", download_mode=DownloadMode.FORCE_REDOWNLOAD)
for subset in iter(dataset):
    if subset[0] == 'test':
        image_folder = os.listdir(subset[1])[0]
        full_folder = os.path.join(subset[1], image_folder)
        for i, image in enumerate(os.listdir(full_folder)):
            if i > 10:
                break
            result = realtime_detector(os.path.join(full_folder, image))
            # 输出结果
            print(result)

```

### 模型局限性以及可能的偏差
  - 对于小物体通用场景会存在检出效果差的情况，建议对过小检出目标的进行限制。
  - 目前模型仅限于pipeline调用，尚未支持Finetune和Evaluation。
  - 复杂专用场景性能会产生显著下降，如复杂视角、超低光照以及严重遮挡等。
  - 当前版本在python 3.7环境测试通过，其他环境下可用性待测试。

## 训练数据介绍
<img src="res/coco-logo.png" width="1000">

<img src="res/coco-examples.jpeg" width="1000">

本模型基于COCO数据集的目标检测部分数据及标注进行训练。COCO数据集的全称是[Microsoft Common Objects in Context](https://cocodataset.org/#home)， 是一个评估计算机视觉模型性能的“黄金”标准基准数据集，旨在推动目标检测、实例分割、看图说话和人物关键点方面的研究。其中目标检测有90个日常常见类别，在学术研究中常用其中的80类作为基准的评测数据集。

## 模型训练流程

模型在线训练暂不支持。部分关键训练细节如下：
- 使用 SGD 优化算法，cos LR scheduler，warmup策略。
- 训练迭代为 300 epoch，其中最后15个epoch关闭数据增强。
- Mosaic，颜色增强等策略被应用到训练预处理中。

## 输入预处理

- 输入图像根据长边resize到416后，padding 为416x416的矩形进行推理
- 图像归一化

## 数据评估及结果
|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/nano.py) |416  |25.8  | 0.91 |1.08 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) |


## 引用

如您的相关著作、作品使用了该模型，请引用以下信息：

```
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```