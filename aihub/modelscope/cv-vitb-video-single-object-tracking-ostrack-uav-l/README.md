
<!--- 以下model card模型说明部分，请使用中文提供（除了代码，bibtex等部分） --->

# <OSTrack>单目标跟踪算法模型介绍
对于一个输入视频，只需在第一帧图像中用矩形框指定待跟踪目标，单目跟踪算法将在整个视频帧中持续跟踪该目标，输出跟踪目标在所有图像帧中的矩形框信息。




## 模型描述
<img src="resources/intro.png" width="800" >

本模型是基于OSTrack方案的单目标跟踪框架，使用ViT作为主干网络进行训练，是One-Stream单目标跟踪算法。
本算法将特征抽取和匹配过程通过注意力机制同步进行，获得较高的目标跟踪精度。

## 期望模型使用方式以及适用范围

该模型适用于视频单目标跟踪场景。

### 如何使用模型

- 根据输入待跟踪视频和第一帧图像对应的待跟踪矩形框（x1, y1, x2, y2），可按照代码范例进行模型推理和可视化。

#### Installation
```
conda create -n anti_uav python=3.7
conda activate anti_uav
# pytorch >= 1.3.0
pip install torch==1.8.1+cu102  torchvision==0.9.1+cu102 torchaudio==0.8.1  --extra-index-url https://download.pytorch.org/whl/cu102
git clone -b ostracker https://github.com/ly19965/CVPR_Anti_UAV
cd CVPR_Anti_UAV
pip install -r requirements/tests.txt 
pip install -r requirements/framework.txt
pip install -r requirements/cv.txt 
```

#### 推理代码范例
```python
from modelscope.utils.cv.image_utils import show_video_tracking_result
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model='damo/cv_vitb_video-single-object-tracking_ostrack-uav-l')
video_path = "./data/test/videos/uav.avi"
init_bbox = [302, 317, 390, 384] # the initial object bounding box in the first frame [x1, y1, x2, y2]
result = video_single_object_tracking((video_path, init_bbox))
show_video_tracking_result(video_path, result[OutputKeys.BOXES], "./tracking_result.avi")
print("result is : ", result[OutputKeys.BOXES])
```

#### Multi-GPU模型训练代码范例

```python
import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import json

# Step 1: 数据集准备
train_dataset = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='train')

# Step 2: 相关参数设置
data_root_dir = '/home/ly261666/.cache/modelscope/hub/datasets/ly261666/3rd_Anti-UAV/master/data_files/extracted/7b8a88c5a8f38cced25ee619b96d924c0eea9f033bb57fc160ca2ec004d1ee6f' # 下载的数据集路径
model_id = 'damo/cv_vitb_video-single-object-tracking_ostrack-uav-l'
video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model=model_id) # 下载modelscope上的模型，配置文件
cache_path = '/home/ly261666/.cache/modelscope/hub/damo/cv_vitb_video-single-object-tracking_ostrack-uav-l'# 下载的modelscope模型路径
cfg_file = os.path.join(cache_path, 'configuration.json')

kwargs = dict(
            cfg_file=cfg_file,
            model=model_id, # 使用ostracker_uav模型
            gpu_ids=[  # 指定训练使用的gpu
            0,1,2,3,4,5,6,7
            ],
            batch_size=64, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
            max_epochs=10, # 总的训练epochs
            load_pretrain=False, # 是否载入预训练模型，若为False，则为从头重新训练, 若为True，则加载modelscope上的模型finetune。
            base_lr_per_img=0.001, # 每张图片的学习率，lr=base_lr_per_img*batch_size
            train_image_dir=data_root_dir, # 训练图片路径
            val_image_dir=data_root_dir, # 测试图片路径
            workers_per_gpu=1,
            )


if __name__ == '__main__':
    # Step 3: 开启训练任务
    trainer = build_trainer(
                        name=Trainers.video_single_object_tracking, default_args=kwargs)
        trainer.train()

```

#### Multi-GPU模型训练代码范例 (加载在Got-10K数据上训练的预训练模型)

```python
import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import json

# Step 1: 数据集准备
train_dataset = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='train')

# Step 2: 相关参数设置
data_root_dir = '/home/ly261666/.cache/modelscope/hub/datasets/ly261666/3rd_Anti-UAV/master/data_files/extracted/7b8a88c5a8f38cced25ee619b96d924c0eea9f033bb57fc160ca2ec004d1ee6f' # 下载的数据集路径
model_id = 'damo/cv_vitb_video-single-object-tracking_ostrack-uav-l'
video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model=model_id) # 下载modelscope上的模型，配置文件
cache_path = '/home/ly261666/.cache/modelscope/hub/damo/cv_vitb_video-single-object-tracking_ostrack-uav-l'# 下载的modelscope模型路径
cfg_file = os.path.join(cache_path, 'configuration.json')
pretrain_model_id = 'damo/cv_vitb_video-single-object-tracking_ostrack-l'
video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model=pretrain_model_id) # 从modelscope上下载pretrain model
pretrain_model = '/home/ly261666/.cache/modelscope/hub/damo/cv_vitb_video-single-object-tracking_ostrack-l/pytorch_model.pt'

kwargs = dict(
            cfg_file=cfg_file,
            model=model_id, # 使用ostracker_uav模型
            gpu_ids=[  # 指定训练使用的gpu
            0,1,2,3,4,5,6,7
            ],
            batch_size=64, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
            max_epochs=10, # 总的训练epochs
            load_pretrain=True, # 是否载入预训练模型，若为False，则为从头重新训练, 若为True，则加载modelscope上的模型finetune。
            pretrain_model=pretrain_model,
            base_lr_per_img=0.001, # 每张图片的学习率，lr=base_lr_per_img*batch_size
            train_image_dir=data_root_dir, # 训练图片路径
            val_image_dir=data_root_dir, # 测试图片路径
            workers_per_gpu=1,
            )


if __name__ == '__main__':
    # Step 3: 开启训练任务
    trainer = build_trainer(
                        name=Trainers.video_single_object_tracking, default_args=kwargs)
        trainer.train()
```

#### Multi-GPU DIY自己模型训练

```python
import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import json

# Step 1: 数据集准备
train_dataset = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='train')

# Step 2: 相关参数设置
data_root_dir = '/home/ly261666/.cache/modelscope/hub/datasets/ly261666/3rd_Anti-UAV/master/data_files/extracted/7b8a88c5a8f38cced25ee619b96d924c0eea9f033bb57fc160ca2ec004d1ee6f' # 下载的数据集路径
model_id = 'damo/cv_vitb_video-single-object-tracking_ostrack-uav-l'
video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model=model_id) # 下载modelscope上的模型，配置文件
cache_path = '/home/ly261666/workspace/trcker_baseline' 
cp -r /home/ly261666/.cache/modelscope/hub/damo/cv_vitb_video-single-object-tracking_ostrack-uav-l/* /home/ly261666/workspace/trcker_baseline/
cfg_file = os.path.join(cache_path, 'configuration.json')

kwargs = dict(
            cfg_file=cfg_file,
            model=model_id, # 使用ostracker_uav模型
            gpu_ids=[  # 指定训练使用的gpu
            0,1,2,3,4,5,6,7
            ],
            batch_size=64, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
            max_epochs=10, # 总的训练epochs
            load_pretrain=False, # 是否载入预训练模型，若为False，则为从头重新训练, 若为True，则加载modelscope上的模型finetune。
            cache_path=cache_path,
            base_lr_per_img=0.001, # 每张图片的学习率，lr=base_lr_per_img*batch_size
            train_image_dir=data_root_dir, # 训练图片路径
            val_image_dir=data_root_dir, # 测试图片路径
            workers_per_gpu=1,
            )


if __name__ == '__main__':
    # Step 3: 开启训练任务
    trainer = build_trainer(
                        name=Trainers.video_single_object_tracking, default_args=kwargs)
```
### 模型局限性以及可能的偏差
- 在遮挡严重场景和背景中存在与目标高度相似的物体场景下，目标跟踪精度可能欠佳。
- 建议在有GPU的机器上进行测试，由于硬件精度影响，CPU上的结果会和GPU上的结果略有差异。


### 相关论文以及引用信息
本模型主要参考论文如下：

```BibTeX
@inproceedings{bertinetto2016fully,
      title={Fully-convolutional siamese networks for object tracking},
        author={Bertinetto, Luca and Valmadre, Jack and Henriques, Joao F and Vedaldi, Andrea and Torr, Philip HS},
          booktitle={Computer Vision--ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part II 14},
            pages={850--865},
              year={2016},
                organization={Springer}
}
```
