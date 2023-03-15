# 视频插帧(应用型)介绍
给定一段低帧率视频，模型会返回更流畅的高帧率视频，可支持任意指定帧率输出。

## 模型描述

全链路插帧模型包含光流计算模块和中间帧生成模块。其中光流计算模型复用了RAFT，详见：https://github.com/princeton-vl/RAFT 。中间帧生成模型包含了光流refine、backward warping以及中间帧融合模块。该模型适用于各类低帧率视频增强，用于提升视频的流畅度，消除卡顿现象。和原版视频插帧模型相比，该版模型在支持任意帧率输出的同时，在大运动、重复纹理、台标字幕等困难场景下的插帧效果更好，适合大部分真实场景视频的插帧任务。

模型效果如下：
| <img src="./data/out.gif"  height=160 width=910 alt="result">|
| :-------------------------------------------------------------------------------: |
|           原始视频(左) , 2倍插帧(中) , 4倍插帧(右)                  |

## 期望模型使用方式以及适用范围
本模型主要用于视频帧率转换，提升视频流畅度。用户可以自行尝试不同分辨率视频输入和不同帧率输出下的模型效果。具体调用方式请参考代码示例。 

### 如何使用
在ModelScope框架下，通过调用简单的Pipeline即可使用当前模型，其中，输入需为字典格式，合法键值包括'video'、'interp_ratio'以及'out_fps'。video提供输入视频地址，为必选项；interp_ratio可指定需要的插帧倍率；out_fps可指定输出帧率。两者任选其一输入即可，若同时指定，则以interp_ratio为准，若均不指定，则默认以2倍率插帧。该模型暂仅支持在GPU上进行推理。输入具体代码示例如下：

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

video = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_frame_interpolation_test.mp4'
video_frame_interpolation_practical_pipeline = pipeline(Tasks.video_frame_interpolation, 'damo/cv_raft_video-frame-interpolation_practical')

# example1, use interp_ratio
input_data1 = {'video': video, 'interp_ratio': 4}
result1 = video_frame_interpolation_practical_pipeline(input_data1)[OutputKeys.OUTPUT_VIDEO]
print('pipeline: the output video path is {}'.format(result1))

# example2, use out_fps
input_data2 = {'video': video, 'out_fps': 50}
result2 = video_frame_interpolation_practical_pipeline(input_data2)[OutputKeys.OUTPUT_VIDEO]
print('pipeline: the output video path is {}'.format(result2))

# example3, use default value for 2x interpolation
input_data3 = {'video': video}
result3 = video_frame_interpolation_practical_pipeline(input_data3)[OutputKeys.OUTPUT_VIDEO]
print('pipeline: the output video path is {}'.format(result3))
```

### 模型局限性以及可能的偏差
该模型在公开数据集vimeo_septuplet上进行了预训练，并在内部数据集上进行了finetune。和原版模型相比，该版模型在学术验证集上的指标略有下降。此外，不同的训练数据增强方法以及光流gt会对模型训练结果产生影响，请用户自行评测后决定如何使用。 

## 训练数据介绍
vimeo_septuplet: 经典的视频插帧数据集，训练集包含64612组图片，验证集包含7824组图片，每组图片包含连续7帧448x256图像， 具体数据可以[下载](http://toflow.csail.mit.edu/index.html#triplet)

### 训练
该模型暂不支持训练(finetune)，该功能将在之后的更新中尽快开放

## 模型评估
该模型支持在ucf_101、middlebury、davis公开数据上进行验证，详见右侧的关联数据集。数据评估结果及示例代码如下
| Dataset | PSNR | SSIM | LPIPS |
|:---- |:----    |:---- |:----|		
|ucf_101|32.37|0.966|0.036|
|middlebury|34.44|0.971|0.035|
|davis|27.51|0.888|0.094|
```python
import os
import tempfile
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.video_frame_interpolation import \
    VideoFrameInterpolationDataset

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
model_id = 'damo/cv_raft_video-frame-interpolation_practical'
cache_path = snapshot_download(model_id)
print(cache_path)
config = Config.from_file(
    os.path.join(cache_path, ModelFile.CONFIGURATION))
dataset_val = MsDataset.load(
    'cv_video-frame-interpolation_ValidationDataset',
    namespace='aojie1997',
    subset_name='middlebury',
    split='validation',
    download_mode=DownloadMode.FORCE_REDOWNLOAD)._hf_ds
eval_dataset = VideoFrameInterpolationDataset(dataset_val, config.dataset)
kwargs = dict(
    model=model_id,
    train_dataset=None,
    eval_dataset=eval_dataset,
    work_dir=tmp_dir)
trainer = build_trainer(default_args=kwargs)
metric_values = trainer.evaluate()
print(metric_values)
```

### 相关论文以及引用信息
该模型借鉴了以下论文的思路或代码：
```
@inproceedings{teed2020raft,
  title={Raft: Recurrent all-pairs field transforms for optical flow},
  author={Teed, Zachary and Deng, Jia},
  booktitle={European conference on computer vision},
  pages={402--419},
  year={2020},
  organization={Springer}
}
@inproceedings{qvi_nips19,
	title={Quadratic video interpolation},
	author={Xiangyu Xu and Li Siyao and Wenxiu Sun and Qian Yin and Ming-Hsuan Yang},
	booktitle = {NeurIPS},
	year={2019}
}
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
@inproceedings{lu2022vfiformer,
    title={Video Frame Interpolation with Transformer},
    author={Liying Lu, Ruizheng Wu, Huaijia Lin, Jiangbo Lu, and Jiaya Jia},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022},
}
```