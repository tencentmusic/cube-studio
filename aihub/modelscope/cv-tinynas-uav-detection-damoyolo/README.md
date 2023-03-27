
<!--- 以下model card模型说明部分，请使用中文提供（除了代码，bibtex等部分） --->

# 实时无人机检测模型介绍

<div align="center"><img src="assets/result.png" width="500"></div>
本模型为**高性能热门应用系列检测模型**中的 <u>实时无人机检测模型</u>，基于面向工业落地的高性能检测框架[DAMOYOLO](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary)，其精度和速度超越当前经典的YOLO系列方法。用户使用的时候，仅需要输入一张图像，便可以获得图像中所有无人机的坐标信息，该模型作为CVPR23 Anti-UAV竞赛的baseline model。



## 模型描述
本模型为实时人体检测模型，基于[DAMOYOLO-S模型](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary)，DAMO-YOLO是一个面向工业落地的目标检测框架，兼顾模型速度与精度，其训练的模型效果超越了目前的一众YOLO系列方法，并且仍然保持极高的推理速度。DAMOYOLO与YOLO系列其它经典工作的性能对比如下图所示：

<div align="center"><img src="https://modelscope.cn/api/v1/models/damo/cv_tinynas_uav-detection_damoyolo/repo?Revision=master&FilePath=assets/DAMOYOLO_performance.jpg&View=true" width="500"></div>

DAMOYOLO整体网络结构如下，整体由backbone (MAE-NAS), neck (GFPN), head (ZeroHead)三部分组成，基于"large neck, small head"的设计思想，对低层空间信息和高层语义信息进行更加充分的融合，从而提升模型最终的检测效果。

<div align="center"><img src="https://modelscope.cn/api/v1/models/damo/cv_tinynas_uav-detection_damoyolo/repo?Revision=master&FilePath=assets/DAMOYOLO_architecture.jpg&View=true" width="2000"></div>


## 期望模型使用方式以及适用范围
该模型适用于无人机检测，输入任意的图像，输出图像中无人机的外接矩形框坐标信息（支持图片中有多个无人机）。

| 类别ID | 类别名称 |
| ------------ | ------------ |
| 1 | uav |

### 如何使用
在ModelScope框架上，提供输入图片，即可以通过简单的Pipeline调用使用当前模型，得到图像中所有无人机的外接矩形框坐标信息。

### Installation
```
conda create -n anti_uav python=3.7
conda activate anti_uav
# pytorch >= 1.3.0
pip install torch==1.8.1+cu102  torchvision==0.9.1+cu102 torchaudio==0.8.1  --extra-index-url https://download.pytorch.org/whl/cu102
git clone https://github.com/ly19965/CVPR_Anti_UAV
cd CVPR_Anti_UAV
pip install -r requirements/tests.txt 
pip install -r requirements/framework.txt
pip install -r requirements/cv.txt 
```

#### 推理代码范例
基础示例代码。下面的示例代码展示的是如何通过一张图片作为输入，得到图片对应的人体坐标信息。
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_tinynas_uav-detection_damoyolo'
input_location = 'data/test/images/uav_detection.jpg'

uav_detection = pipeline(Tasks.domain_specific_object_detection, model=model_id)
result = uav_detection(input_location)
print("result is : ", result)
```

#### Multi-GPU模型训练代码范例
```python
import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import os
import json

# Step 1: 数据集准备，可以使用modelscope上已有的数据集，也可以自己在本地构建COCO数据集
train_dataset = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='train')#, download_mode=DownloadMode.FORCE_REDOWNLOAD)
val_dataset = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='validation')#, download_mode=DownloadMode.FORCE_REDOWNLOAD)

# Step 2: 相关参数设置
data_root_dir = '/home/ly261666/.cache/modelscope/hub/datasets/ly261666/3rd_Anti-UAV/master/data_files/extracted/7b8a88c5a8f38cced25ee619b96d924c0eea9f033bb57fc160ca2ec004d1ee6f'
train_img_dir = osp.join(data_root_dir, 'train')
val_img_dir = osp.join(data_root_dir, 'validation')
train_anno_path = osp.join(data_root_dir, 'train.json')
val_anno_path = osp.join(data_root_dir, 'validation.json')
val_anno_path = '/home/ly261666/workspace/maas/modelscope_project/Mass_env/or_data/train_data/validation.json'
cache_path = '/home/ly261666/.cache/modelscope/hub/damo/cv_tinynas_uav-detection_damoyolo'

kwargs = dict(
            cfg_file=os.path.join(cache_path, 'configuration.json'),
            model='damo/cv_tinynas_uav-detection_damoyolo', # 使用DAMO-YOLO-S模型 
            gpu_ids=[  # 指定训练使用的gpu
            0,1,2,3,4,5,6,7
            ],
            batch_size=64, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
            max_epochs=10, # 总的训练epochs
            num_classes=1, # 自定义数据中的类别数
            cache_path=cache_path,
            load_pretrain=False, # 是否载入预训练模型，若为False，则为从头重新训练
            base_lr_per_img=0.01, # 每张图片的学习率，lr=base_lr_per_img*batch_size
            train_image_dir=train_img_dir, # 训练图片路径
            val_image_dir=val_img_dir, # 测试图片路径
            train_ann=train_anno_path, # 训练标注文件路径
            val_ann=val_anno_path, # 测试标注文件路径
            )

# Step 3: 开启训练任务
if __name__ == '__main__':
    trainer = build_trainer(
                        name=Trainers.tinynas_damoyolo, default_args=kwargs)
    trainer.train()
```

#### Multi-GPU模型训练代码范例 (加载在COCO数据上训练的预训练模型)
```python
import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import os
import json

# Step 1: 数据集准备，可以使用modelscope上已有的数据集，也可以自己在本地构建COCO数据集
train_dataset = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='train')#, download_mode=DownloadMode.FORCE_REDOWNLOAD)
val_dataset = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='validation')#, download_mode=DownloadMode.FORCE_REDOWNLOAD)

# Step 2: 相关参数设置
data_root_dir = '/home/ly261666/.cache/modelscope/hub/datasets/ly261666/3rd_Anti-UAV/master/data_files/extracted/7b8a88c5a8f38cced25ee619b96d924c0eea9f033bb57fc160ca2ec004d1ee6f'
train_img_dir = osp.join(data_root_dir, 'train')
val_img_dir = osp.join(data_root_dir, 'validation')
train_anno_path = osp.join(data_root_dir, 'train.json')
val_anno_path = osp.join(data_root_dir, 'validation.json')
val_anno_path = '/home/ly261666/workspace/maas/modelscope_project/Mass_env/or_data/train_data/validation.json'
cache_path = '/home/ly261666/.cache/modelscope/hub/damo/cv_tinynas_uav-detection_damoyolo'
pretrain_model = '/home/ly261666/.cache/modelscope/hub/damo/cv_tinynas_object-detection_damoyolo/damoyolo_s.pt'# 下载的modelscope预训练模型路径

kwargs = dict(
            cfg_file=os.path.join(cache_path, 'configuration.json'),
            model='damo/cv_tinynas_uav-detection_damoyolo', # 使用DAMO-YOLO-S模型 
            gpu_ids=[  # 指定训练使用的gpu
            0,1,2,3,4,5,6,7
            ],
            batch_size=64, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
            max_epochs=10, # 总的训练epochs
            num_classes=1, # 自定义数据中的类别数
            cache_path=cache_path,
            load_pretrain=True, # 是否载入预训练模型，若为False，则为从头重新训练
            pretrain_model=pretrain_model, # 加载pretrain_model继续训练 
            base_lr_per_img=0.01, # 每张图片的学习率，lr=base_lr_per_img*batch_size
            train_image_dir=train_img_dir, # 训练图片路径
            val_image_dir=val_img_dir, # 测试图片路径
            train_ann=train_anno_path, # 训练标注文件路径
            val_ann=val_anno_path, # 测试标注文件路径
            )

# Step 3: 开启训练任务
if __name__ == '__main__':
    trainer = build_trainer(
                        name=Trainers.tinynas_damoyolo, default_args=kwargs)
    trainer.train()
```

#### Multi-GPU训练自己DIY模型
```python
import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import os
import json

# Step 1: 数据集准备，可以使用modelscope上已有的数据集，也可以自己在本地构建COCO数据集
train_dataset = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='train')#, download_mode=DownloadMode.FORCE_REDOWNLOAD)
val_dataset = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='validation')#, download_mode=DownloadMode.FORCE_REDOWNLOAD)

# Step 2: 相关参数设置
data_root_dir = '/home/ly261666/.cache/modelscope/hub/datasets/ly261666/3rd_Anti-UAV/master/data_files/extracted/7b8a88c5a8f38cced25ee619b96d924c0eea9f033bb57fc160ca2ec004d1ee6f'
train_img_dir = osp.join(data_root_dir, 'train')
val_img_dir = osp.join(data_root_dir, 'validation')
train_anno_path = osp.join(data_root_dir, 'train.json')
val_anno_path = osp.join(data_root_dir, 'validation.json')
val_anno_path = '/home/ly261666/workspace/maas/modelscope_project/Mass_env/or_data/train_data/validation.json'
cache_path = '/home/ly261666/workspace/det_baseline'

kwargs = dict(
            cfg_file=os.path.join(cache_path, 'configuration.json'),
            model='damo/cv_tinynas_uav-detection_damoyolo', # 使用DAMO-YOLO-S模型 
            gpu_ids=[  # 指定训练使用的gpu
            0,1,2,3,4,5,6,7
            ],
            batch_size=64, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
            max_epochs=10, # 总的训练epochs
            num_classes=1, # 自定义数据中的类别数
            cache_path=cache_path,
            load_pretrain=False, # 是否载入预训练模型，若为False，则为从头重新训练
            cache_path=cache_path, # DIY模型目录，包含configuration.json文件即可
            base_lr_per_img=0.01, # 每张图片的学习率，lr=base_lr_per_img*batch_size
            train_image_dir=train_img_dir, # 训练图片路径
            val_image_dir=val_img_dir, # 测试图片路径
            train_ann=train_anno_path, # 训练标注文件路径
            val_ann=val_anno_path, # 测试标注文件路径
            )

# Step 3: 开启训练任务
if __name__ == '__main__':
    trainer = build_trainer(
                        name=Trainers.tinynas_damoyolo, default_args=kwargs)
    trainer.train()
```

## 训练数据介绍
- [cvpr2023_UAV-Anti_train](https://modelscope.cn/datasets/ly261666/3rd_Anti-UAV/summary)


### 相关论文以及引用信息
本模型主要参考论文如下（论文链接：[link](https://arxiv.org/abs/2211.15444)）：

```BibTeX
 @article{damoyolo,
  title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
  author={Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang and Xiuyu Sun},
  journal={arXiv preprint arXiv:2211.15444v2},
  year={2022}
}
```
