
# 视频摘要
输入一段长视频和一段文字描述，算法根据用户输入的文字对输入视频中的相关片段进行自适应的视频摘要，根据帧号可以合成一段短视频（摘要视频）。

### 模型结构：

<p align="left">
    <img src="video/framework.png" alt="donuts" />
    
如上图所示，用户输入的文字使用CLIP的文本编码模型提取特征，视频帧使用CLIP的图像编码模型提取特征，两者构造一个Language-Guided Attention，得到Attented Feature,经过一个Transfer模型对每帧图像进行打分，得到最后的摘要视频。

### 如何使用
输入视频目前只支持.mp4格式。

在ModelScope框架上，提供一段长视频和N句英文的文字描述，即可以通过简单的Pipeline调用来使用视频摘要模型。

如果不提供文字描述，算法则会使用根据视频生成的文字作为输入。

#### 代码范例
```python
import shutil
import os
import tempfile
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

video_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_category_test_video.mp4'
# input can be sentences such as input_sentences=['phone', 'hand'], or input_sentences=None
input_sentences = ['phone', 'hand']

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

model_id = 'damo/cv_clip-it_video-summarization_language-guided_en'
summarization_pipeline = pipeline(Tasks.language_guided_video_summarization, model=model_id,
    tmp_dir=tmp_dir)
result = summarization_pipeline((video_path, input_sentences))

print(f'video summarization output: \n{result}.')

```
### 模型局限性以及可能的偏差
模型在TvSum数据集训练，对于实际场景的数据和用户偏好需要使用数据进行finetue。

## 训练数据介绍
训练数据为[TvSum](https://github.com/yalesong/tvsum)公开数据集。由于license的原因，此处不便于上传数据集，用户可以前往自行下载。

## 模型训练流程

```python
import os
import shutil
import tempfile
import unittest

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.cv.language_guided_video_summarization import ClipItVideoSummarization
from modelscope.msdatasets.task_datasets import LanguageGuidedVideoSummarizationDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

model_id = 'damo/cv_clip-it_video-summarization_language-guided_en'
cache_path = snapshot_download(model_id)
config = Config.from_file(os.path.join(cache_path, ModelFile.CONFIGURATION))
dataset_train = LanguageGuidedVideoSummarizationDataset('train', config.dataset, cache_path)
dataset_val = LanguageGuidedVideoSummarizationDataset('test', config.dataset, cache_path)

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
PYTHONPATH=. python tests/run.py --pattern test_language_guided_video_summarization_trainer.py
```

## 数据评估及结果
评估指标：F-Score

|   数据集   | TVSum |
|:-------:|:-----:|
| PGL-SUM | 65.75 |

### 相关论文以及引用信息

```BibTeX
@INPROCEEDINGS{
    author    = {Narasimhan M, Rohrbach A, Darrell T},
    title     = {CLIP-It! language-guided video summarization},
    booktitle = {Advances in Neural Information Processing Systems},
    month     = {December},
    year      = {2021},
    pages     = {13988-14000}
}
```
