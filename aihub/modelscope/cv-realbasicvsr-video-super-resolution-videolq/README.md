
# Investigating Tradeoffs in Real-World Video Super-Resolution

## 模型描述
RealBasicVSR提出了一个预清理模块，其可以在传播之前抑制退化。在许多具有挑战性的情况下，将图像单次输入到预清理模块并不能有效地消除过度退化。一种简单而有效的方法是将图像多次输入预清理，以进一步抑制退化。通过动态优化方案，清理阶段会自动停止以避免过度平滑。这项工作采用了BasicVSR作为超分网络，因为它通过长期传播在非盲超分辨率中具有良好的性能，并且结构简单。

<img src='./data/RealBasicVSR.png' width=640 alt="Overview of RealBasicVSR">

## 期望模型使用方式以及适用范围
本模型使用于一般视频超分辨率。
### 如何使用
在ModelScope框架上，提供输入视频，即可通过简单的Pipeline调用来使用。
#### 代码范例
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/000.mp4'
video_super_resolution_pipeline = pipeline(
    Tasks.video_super_resolution,
    'damo/cv_realbasicvsr_video-super-resolution_videolq')
result = video_super_resolution_pipeline(video)[OutputKeys.OUTPUT_VIDEO]
```

### 模型局限性以及可能的偏差
模型对于大部分真实场景效果良好，对于小部分降质十分严重的情况可能表现不佳。
## 测试数据介绍
VideoLQ

文件类型：.PNG

文件数量：50个视频片段

内容：每个视频片段包含100帧视频帧（除`030`、`031`、`032`、`033`外）

## 数据评估及结果
| name         | Dataset | NIQE   |
|:-------------| :---- |:-------|
| RealBasicVSR | VideoLQ | 2.5693 |

```python
import os
import tempfile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.msdatasets.task_datasets.video_super_resolution import \
    VideoSuperResolutionDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode
from collections import Counter
import numpy as np

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
model_id = 'damo/cv_realbasicvsr_video-super-resolution_videolq'
cache_path = snapshot_download(model_id)
dataset_test = MsDataset.load(
    'VideoLQ',
    namespace='huizheng',
    subset_name='default',
    split='test',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)._hf_ds

clip_num_nframes = Counter(dataset_test['Clip Num'])
indices = np.cumsum([0] + list(clip_num_nframes.values()))
dataset = []
for index in range(len(indices) - 1):
    sub_test_dataset = []
    for frame in range(indices[index], indices[index + 1]):
        sub_test_dataset.append({'LQ Frame:FILE': dataset_test[frame]['LQ Frame:FILE'], 'Clip Num': 0})
    dataset.append(sub_test_dataset)

test_dataset = VideoSuperResolutionDataset(dataset[0])  # the first clip, 100 frames
kwargs = dict(
    model=model_id,
    train_dataset=None,
    eval_dataset=test_dataset,
    work_dir=tmp_dir)
trainer = build_trainer(default_args=kwargs)
metric_values = trainer.evaluate()
print(metric_values)
```
### 相关论文以及引用信息
如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```
@inproceedings{chan2022investigating,
  author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title = {Investigating Tradeoffs in Real-World Video Super-Resolution},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2022}
}
```