
# PGL_SUM视频摘要-Web视频领域
输入一段长视频，算法对视频进行镜头切割得到视频片段，评估视频帧的重要性，输出重要视频帧的帧号，根据帧号可以合成一段短视频（摘要视频）。
<!--


输入视频：

[![IMAGE ALT TEXT](video/preview_input_4.jpg)](https://dmshared.oss-cn-hangzhou.aliyuncs.com/james.wjg/MAAS/video/video_category_test_video.mp4?OSSAccessKeyId=LTAI5tC7NViXtQKpxFUpxd3a&Expires=2023833695&Signature=9ZUwnqVyGdMlmukFazup0r9%2BbJ8%3D)


输出摘要视频：

[![IMAGE ALT TEXT](video/preview_output_4.jpg)](https://dmshared.oss-cn-hangzhou.aliyuncs.com/james.wjg/MAAS/video/summarization_result.mp4?OSSAccessKeyId=LTAI5tC7NViXtQKpxFUpxd3a&Expires=2023835215&Signature=62dIjYT%2BjTkCiNbsSNY6u%2FicuYk%3D)
-->



### 模型结构：

<p align="left">
    <img src="video/framework.png" alt="donuts" />
    
如上图所示，采用local和global的multi head attention构成的transformer模型

### 如何使用
在ModelScope框架上，提供一段长视频，即可以通过简单的Pipeline调用来使用视频摘要模型。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_category_test_video.mp4'
summarization_pipeline = pipeline(Tasks.video_summarization, model='damo/cv_googlenet_pgl-video-summarization')
result = summarization_pipeline(video_path)
print(f'video summarization output: {result}.')
```
### 模型局限性以及可能的偏差
模型在TvSum数据集训练，对于实际场景的数据和用户偏好需要使用数据进行finetue。

## 训练数据介绍
训练数据为[TvSum](https://github.com/yalesong/tvsum)公开数据集。

由于license的限制，此处没有上传原始视频数据，如需原始数据，请在TvSum的链接中自行下载。在data/eccv16_dataset_tvsum_google_pool5.h5文件中有提供从原始视频中使用googlenet提取的特征，可以通过该特征文件直接训练并得到测试的F-Score指标。


## 模型训练流程
```python
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.cv.video_summarization import PGLVideoSummarization
from modelscope.msdatasets.task_datasets import VideoSummarizationDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

model_id = 'damo/cv_googlenet_pgl-video-summarization'
cache_path = snapshot_download(model_id)
config = Config.from_file(os.path.join(cache_path, ModelFile.CONFIGURATION))
dataset_train = VideoSummarizationDataset('train', config.dataset, cache_path)
dataset_val = VideoSummarizationDataset('test', config.dataset, cache_path)

kwargs = dict(
    model=model_id,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    work_dir=tmp_dir)
trainer = build_trainer(default_args=kwargs)
trainer.train()
results_files = os.listdir(tmp_dir)
```

也可通过unitest代码直接调取

```shell
PYTHONPATH=. python tests/run.py --pattern test_video_summarization_trainer.py
```

用户的视频转换为训练数据请参考[PGL-SUM](https://github.com/e-apostolidis/PGL-SUM)。

## 数据评估及结果

以上模型训练流程中已包含了测试集的评估，训练过程中能够直接看到测试集的评估指标

评估指标：F-Score

|   数据集   | TVSum |
|:-------:| :----: |
| PGL-SUM | 61.0 |

### 相关论文以及引用信息

```BibTeX
@INPROCEEDINGS{9666088,
    author    = {Apostolidis, Evlampios and Balaouras, Georgios and Mezaris, Vasileios and Patras, Ioannis},
    title     = {Combining Global and Local Attention with Positional Encoding for Video Summarization},
    booktitle = {2021 IEEE International Symposium on Multimedia (ISM)},
    month     = {December},
    year      = {2021},
    pages     = {226-234}
}
```
