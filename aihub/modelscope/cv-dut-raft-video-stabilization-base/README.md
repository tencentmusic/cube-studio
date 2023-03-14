# DUT-RAFT 视频稳像模型
该模型为抖动视频稳像模型，输入一个抖动视频，实现端到端的视频稳像（视频去抖动），返回稳像处理后的稳定视频。

模型效果如下，Demo中的测试视频源来自[DVS](https://github.com/googleinterns/deep-stabilization)开源数据集。
| <img src="./img/stab_demo.gif">|
| :-------------------------------------------------------------------------------: |
|           原始视频(左) , 去抖视频(右)                  |

## 模型描述

1. 该模型基于DUT视频稳像模型进行修改；DUT包含一个关键点检测模块，一个基于网格轨迹估计的运动传播模块，以及一个动态轨迹平滑模块。 DUT模型是无监督的，只需要不稳定视频进行训练。
2. 相比于DUT原文，该模型将光流估计模块由PWCNet替换为较新的RAFT，并调整部分超参重新训练。同时，该模型更改了warp逻辑，使得稳像后的视频尽可能维持原始画质。
3. 视频稳像后可能会在视频边缘处出现内容缺失（黑边），因此，本模型对输出视频进行了裁剪，再缩放至原视频尺寸。

![image](./img/NetworkStructure.png)

## 模型期望使用方式和适用范围

### 适用范围
1. 该模型适用于多种格式的视频输入，给定抖动视频，生成稳像后的稳定视频；
2. 需要注意的是，如果输入视频包含镜头切换，运动轨迹估计将出现错误，导致错误的稳像结果，因此建议输入单一镜头的抖动视频；
3. 建议输入横屏视频，由于训练数据中未包含竖屏视频，本算法模型对于竖屏输入（宽<高）的视频稳像表现可能不佳；
4. 使用16G显存的显卡测试时，建议的最大输入为 30fps帧率下30s时长的1920x1080分辨率视频。

### 如何使用
在 ModelScope 框架上，提供输入视频，即可以通过简单的 Pipeline 调用来使用视频稳像模型。模型暂时仅支持在GPU上进行推理，具体示例代码如下：

#### 推理代码范例
```python
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

test_video = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_stabilization_test_video.avi'
video_stabilization = pipeline(Tasks.video_stabilization, 
                       model='damo/cv_dut-raft_video-stabilization_base')
out_video_path = video_stabilization(test_video)[OutputKeys.OUTPUT_VIDEO]
print('Pipeline: the output video path is {}'.format(out_video_path))
```

#### 数据集评测
利用NUS视频稳像数据集进行评测
```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.video_stabilization import \
    VideoStabilizationDataset

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

model_id = 'damo/cv_dut-raft_video-stabilization_base'
cache_path = snapshot_download(model_id)
config = Config.from_file(os.path.join(cache_path, ModelFile.CONFIGURATION))

dataset_val = MsDataset.load(
    'NUS_video-stabilization',
    namespace='zcmaas',
    subset_name='Regular',
    split='train',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds
eval_dataset = VideoStabilizationDataset(dataset_val, config.dataset)
kwargs = dict(
    model=model_id,
    train_dataset=None,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir)

trainer = build_trainer(default_args=kwargs)
metric_values = trainer.evaluate()

print(metric_values)
```
### 模型局限性以及可能的偏差

- 由于训练数据中未包含竖屏视频，本算法模型对于竖屏输入（宽<高）的视频稳像表现可能不佳；
- 对于快速场景切换的视频输入，本算法模型可能表现不佳。

## 训练数据介绍
模型使用公开数据集[DeepStab](http://cg.cs.tsinghua.edu.cn/download/DeepStab.zip)进行训练。

## 说明与引用
本算法模型的训练与推理过程参考了一些开源项目：

- 训练与推理部分代码参考自[DUTCode](https://github.com/Annbless/DUTCode)；
- 光流估计部分代码参考自[RAFT](https://github.com/princeton-vl/RAFT)；
- Metric评测代码参考自[DIFRINT](https://github.com/jinsc37/DIFRINT)；

如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```
@article{xu2022dut,
  title={Dut: Learning video stabilization by simply watching unstable videos},
  author={Xu, Yufei and Zhang, Jing and Maybank, Stephen J and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={4306--4320},
  year={2022},
  publisher={IEEE}
}

@inproceedings{teed2020raft,
  title={Raft: Recurrent all-pairs field transforms for optical flow},
  author={Teed, Zachary and Deng, Jia},
  booktitle={European conference on computer vision},
  pages={402--419},
  year={2020},
  organization={Springer}
}

@article{Choi_TOG20,
	author = {Choi, Jinsoo and Kweon, In So},
	title = {Deep Iterative Frame Interpolation for Full-Frame Video Stabilization},
	year = {2020},
	issue_date = {February 2020},
	publisher = {Association for Computing Machinery},
	volume = {39},
	number = {1},
	issn = {0730-0301},
	url = {https://doi.org/10.1145/3363550},
	journal = {ACM Transactions on Graphics},
	articleno = {4},
	numpages = {9},
}
```
