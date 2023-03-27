# DEPE模型介绍
基于多摄像头的纯视觉3D目标检测方法在自动驾驶领域得到越来越广泛的关注。DEPE模型采用Transformer的end-to-end的结构设计，无需传统检测方法中手工设计的要素(如anchor，NMS等), 而是直接利用图片的2D特征进行时(历史帧)空(多视角)融合，并在训练过程中利用LiDAR点云监督图片depth的预测进而改进3D感知精度，最后经过DETR-like的transformer decoder输出3D目标。在nuScenes数据集上的演示demo如下：

![demo](description/vis-5fps.gif)


## 模型描述

DEPE模型以PETRv2([论文](https://arxiv.org/abs/2206.01256), [代码](https://github.com/megvii-research/PETR))方法为基础，该方法的核心思路是将图片在depth方向采样并结合相机参数生成3D坐标，将3D的位置编码与2D特征结合(3DPE)，从而进行3D检测。DEPE方法扩展了预测图片像素depth的方式，借鉴BEVDepth([论文](https://arxiv.org/abs/2206.10092), [代码](https://github.com/Megvii-BaseDetection/BEVDepth))提出的方法，借助LiDAR点云数据对图片depth的预测进行监督，将depthmap编码后与3DPE结合，以提升目标在3D场景的感知精度。该方式简洁有效，在nuScenes-val数据集上，输入分辨率800x320时，NDS指标相比PETRV2有0.52%的提升（**51.07%-->51.59%**）。DEPE方法的整体框架如下图所示：

![depe](https://modelscope.cn/api/v1/models/damo/cv_object-detection-3d_depe/repo?Revision=master&FilePath=description/depe.jpg&View=true)

## 期望模型使用方式以及适用范围

本模型适用于自动驾驶场景，支持nuScenes数据集的格式作为输入，主要以多视角的视觉输入为主，预测3D场景的10类目标(如行人、车辆)的位置、尺寸、朝向、速度、属性等信息。

### 如何使用

本模型支持nuScenes数据集的推理，用户需要对原始数据集进行处理，并按照预定格式放置于特定文件夹下。首先请参考mmdet3d的官方[文档](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md)对原始数据进行预处理(注：mmdet3d在1.0后重构了坐标系定义，推理代码支持mmdet3d<1.0和mmdet3d>=1.0的版本，但建议使用最新版本)，然后使用PETRv2的官方[工具](https://github.com/megvii-research/PETR/blob/main/tools/generate_sweep_pkl.py)脚本生成pkl文件。默认情况下，模型以nuScenes-mini的val数据集演示模型推理效果，该数据集包含2个场景共81帧。

#### 代码范例
<!--- 本session里的python代码段，将被ModelScope模型页面解析为快速开始范例--->
```python
import cv2
import os
import os.path as osp
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.msdatasets import MsDataset

# use nuScenes-mini dataset as input
ms_ds_nuscenes = MsDataset.load('nuScenes_mini', namespace='shaoxuan')
data_path = ms_ds_nuscenes.config_kwargs['split_config']
val_dir = data_path['validation']
val_root = val_dir + '/' + os.listdir(val_dir)[0] + '/'
# pipeline
depe = pipeline(Tasks.object_detection_3d, model='damo/cv_object-detection-3d_depe')
sample_idx = 10  # set sample index in [0,80]
input_dict = {'data_root': val_root, 'sample_idx': sample_idx}
result = depe(input_dict)
rst_img = result[OutputKeys.OUTPUT_IMG]
if result is not None:
    cv2.imwrite('result.jpg', rst_img)
    print(f'Output written to {osp.abspath("https://modelscope.cn/api/v1/models/damo/cv_object-detection-3d_depe/repo?Revision=master&FilePath=result.jpg&View=true")}')
# if you want to show the result, you can run
import matplotlib.pyplot as plt
plt.axis('off')
plt.imshow(rst_img)
```

### 模型局限性以及可能的偏差
1. 受限于模型训练数据的规模，在丰富度和多样性上与真实场景存在一定偏差，影响模型在真实场景的效果
2. 模型的参数量较大，在车机端落地有较大改进空间
3. 目前尚未支持模型的finetune

## 训练数据介绍
本模型使用的训练数据为nuScenes，是自动驾驶领域公开的大规模数据集之一，共采集了1000个场景，每个场景约20秒，涵盖了复杂的交通状况和天气、光照变化。数据采集的设备包括6个相机、1个LIDAR，5个RADAR，以及GPS和IMU。详见：[官网](https://www.nuscenes.org/nuscenes#download)

## 模型训练流程
DEPE模型训练的图片输入分辨率为800x320，使用v2-99的预训练模型作为backbone, 初始学习率2e-4，采用cosine退火策略训练24epoch，单卡batch_size为1，使用8卡v100耗时18小时。


## 数据评估及结果
DEPE模型与原始PETRv2模型在nuScenes-val的指标对比如下：
| Name | mAP↑ | mATE↓ | mASE↓ | mAOE↓ | mAVE↓ | mAAE↓ | NDS↑ |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ | ------------ |
| PETRv2_p4_800x320_DN | 0.4191 | 0.7005 | 0.2632 | 0.4555 | 0.3814 | 0.1878 | 0.5107 |
| DEPE_p4_800x320_DN | 0.4253 | 0.7031 | 0.2681 | 0.4320 | 0.3791 | 0.1874 | 0.5159 |

逐个类别的指标如下：
| Object Class | AP | ATE(m) | ASE(1-IOU) | AOE(rad) | AVE(m/s) | AAE(1-acc) |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ |
| car | 0.6103 | 0.4762 | 0.1464 | 0.0742 | 0.3233 | 0.1914 |
| truck | 0.3937 | 0.6865 | 0.2001 | 0.1023 | 0.3356 | 0.2046 |
| bus | 0.4687 | 0.7130 | 0.1916 |  0.0824 | 0.7187 | 0.2394 |
| trailer | 0.2391 | 1.0700 | 0.2495 | 0.6014 | 0.2770 | 0.1486 |
| construction_vehicle | 0.1446 | 1.0407 | 0.4767 | 1.0548 | 0.1281 | 0.3404 |
| pedestrian | 0.5100 | 0.6435 | 0.2900 | 0.5347 | 0.4218 | 0.1811 |
| motorcycle | 0.4091 | 0.6550 | 0.2623 | 0.5569 | 0.5812 | 0.1804 |
| bicycle | 0.4129 | 0.5723 | 0.2551 | 0.7236 | 0.2474 | 0.0132 |
| traffic_cone | 0.5660 | 0.5150 | 0.3178 | nan | nan | nan |
| barrier | 0.4985 | 0.6407 | 0.2914 | 0.1573 | nan | nan |



### 相关论文以及引用信息
该项目的部分代码来自于[PETRv2](https://github.com/megvii-research/PETR)， [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)，[BEVDet](https://github.com/HuangJunJie2017/BEVDet)，非常感谢他们开源了相关工作。
如果你觉得该模型对你有所帮助，请考虑引用下面的相关的论文：
```
@article{liu2022petrv2,
  title={PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images},
  author={Liu, Yingfei and Yan, Junjie and Jia, Fan and Li, Shuailin and Gao, Qi and Wang, Tiancai and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2206.01256},
  year={2022}
}
```
```
@article{li2022bevdepth,
  title={BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection},
  author={Li, Yinhao and Ge, Zheng and Yu, Guanyi and Yang, Jinrong and Wang, Zengran and Shi, Yukang and Sun, Jianjian and Li, Zeming},
  journal={arXiv preprint arXiv:2206.10092},
  year={2022}
}
```
```
@article{huang2022bevdet4d,
  title={BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection},
  author={Huang, Junjie and Huang, Guan},
  journal={arXiv preprint arXiv:2203.17054},
  year={2022}
}
```