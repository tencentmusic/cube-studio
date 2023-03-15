

# 手部2D关键点检测模型介绍

输入一张手部图像，实现端到端的手部关键点检测，输出完整的手部21个关键点。

<div align="center">
  <img src="assets/hand_21_landmark.jpeg" width="500" />
</div>

## 模型描述

该模型采用自顶向下的Heatmap手部关键点检测框架，通过端对端的快速推理可以得到图像中的全部手部关键点。
本模型基于HRNetv2和DarkPose方法，架构如下：
<div align="center">
  <img src="./assets/DarkPose1.jpg" width="800" />
  <center>DarkPose</center>
</div>

基于heatmap的关键点检测方法一般存在一个分辨率变化的编码-解码过程，即原始图片分辨率较大，需要先缩小后才能输入模型进行预测，得到预测结果后再还原到输入图片的原始分辨率，再得到预测坐标。这个过程可能会引入一些误差：

- 编码过程：GT在缩小后，渲染得到的高斯热图可能存在误差；
- 解码过程：预测的heatmap在还原到原始分辨率后，得到的坐标也可能存在误差

本文假设预测的heatmap是满足二维高斯分布的，因此基于heatmap的一阶导数和二阶导数来计算偏移方向。但在实验中发现，预测的heatmap并不严格遵守高斯分布，可能会出现多峰的情况，因此增加了一个步骤，对预测的heatmap进行平滑处理，使结果符合假设，从而保证解码效果的准确性。

## 期望模型使用方式以及适用范围

使用方式：
- 在手部图像上进行端到端推理

使用范围：
- 包含手部的单张RGB图像

目标场景：
- 手势姿态估计
- 手势驱动游戏

### 如何使用

在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用来完成手部2D关键点检测任务。

*注意*：如果使用notebook运行demo代码或者下载代码仓库到本地运行，需要安装numpy==1.21.5。

#### 代码范例

```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

model_id = 'damo/cv_hrnetw18_hand-pose-keypoints_coco-wholebody'
hand_2d_keypoints = pipeline(Tasks.hand_2d_keypoints, model=model_id)
result = hand_2d_keypoints('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/hand_keypoints.jpg')
print(result)
```

### 模型局限性以及可能的偏差

**注意：** 当前手部检测模型是采用的[mmdet](https://github.com/open-mmlab/mmpose/blob/master/demo/docs/mmdet_modelzoo.md)提供的基于OneHand数据集训练的手部目标检测模型，对于复杂图像或单张图像中有多个手部的情况可能效果不佳。后续将更新更优的手部检测模型以达到更好的效果。

## 训练数据介绍

手部关键点检测模型是使用[COCO-Wholebody数据集](https://github.com/jin-s13/COCO-WholeBody/)进行训练的。

```
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## 模型训练流程
自动下载和使用托管在modelscope DatasetHub上的数据集：

```python
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode

model_id = 'damo/cv_hrnetw18_hand-pose-keypoints_coco-wholebody'
cfg_options = {'train.max_epochs': 210}
work_dir = "./output"
trainer_name = Trainers.easycv

train_dataset = MsDataset.load(
    dataset_name='cv_hand_2d_keypoints_coco_wholebody',
    namespace='chenhyer',
    split='train',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
eval_dataset = MsDataset.load(
    dataset_name='cv_hand_2d_keypoints_coco_wholebody',
    namespace='chenhyer',
    split='validation',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=work_dir,
    cfg_options=cfg_options)

print("build trainer.")
trainer = build_trainer(trainer_name, kwargs)

print("start training.")
trainer.train()
```
### 训练参数
- 最大迭代次数210
- 优化器：`Adam`, lr=0.0005
- lr_scheduler: step with warmup;

## 模型推理流程
可通过如下代码对模型进行评估验证，我们在modelscope的DatasetHub上存储了`COCO_WholeBody`的验证集，方便用户下载调用。

```python
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode

model_id = 'damo/cv_hrnetw18_hand-pose-keypoints_coco-wholebody'
cfg_options = {'train.max_epochs': 100}
work_dir = "./output"
trainer_name = Trainers.easycv

eval_dataset = MsDataset.load(
    dataset_name='cv_hand_2d_keypoints_coco_wholebody',
    namespace='chenhyer',
    split='subtrain',  # 根据需要设置成需要测试的split，目前该数据集包括3个split: train, subtrain, validation
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

kwargs = dict(
    model=model_id,
    train_dataset=None,
    eval_dataset=eval_dataset,
    work_dir=work_dir,
    cfg_options=cfg_options)

print("build trainer.")
trainer = build_trainer(trainer_name, kwargs)

print("start evaluation.")
metric_values = trainer.evaluate()
print(metric_values)
```


### 预处理

基于手部检测框裁剪图像后，缩放到256×256大小。
本模型设计到的预处理主要有以下几个，具体的组织方式见配置文件：
 - TopDownAffine
 - MMToTensor
 - NormalizeTensor
 - PoseCollect


## 数据评估及结果

COCO-Wholebody数据集上模型指标：

| Method | 输入大小 | PCK | AUC | NME | 
| ------------ | ------------ | ------------ | ------------ | ------------ |
| litehrnet_w18 | 256x256 | 0.8161 | 0.8393 | 4.3899 |

![input](assets/hand.jpg)
![output](assets/hand_result.jpg)

### 相关论文以及引用信息

```
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```

```
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
}
```
