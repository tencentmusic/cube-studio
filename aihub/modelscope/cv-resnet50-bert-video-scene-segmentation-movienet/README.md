
# BaSSL视频场景分割模型介绍
> 视频场景分割任务：场景（scene）定义为一段在语义上具有连续性的视频片段，视频场景分割指的是将一段视频分成若干个场景。

本模型使用基于ResNet-50的结构进行特征提取，而后使用基于bert的结构进行场景分割。


## 模型描述
本模型由shot特征提取器（shot encoder）和上下文关系网络（contextual relation network-CRN）组成。其中shot encoder基于ResNet-50结构，CRN基于bert结构。

本模型使用自监督学习（self-supervised learning）的训练方式，整个训练流程分为两个阶段。其训练流程如下图所示。
在预训练阶段中，整个模型在四个上游任务上进行联合训练，通过这四个任务的联合训练，模型可以在不需要任何人工标注的标签的情况下得到较好的shot encoder。
在微调阶段中，冻结shot encoder的参数，并且利用视频场景分割的数据和标签训练CRN，最终模型可以在MovieNet视频场景分割数据集上达到SOTA性能。

![BaSSL模型训练流程](data/bassl-schema.jpg)

## 期望模型使用方式以及适用范围
本模型主要应用于长视频场景分割领域。



### 如何使用
输入一段长视频，模型会先对该视频进行镜头（shot）分割，然后再对分割出来的镜头进行聚合并输出最终的场景结果。
模型的输出包括分割得到的场景数目、每个场景包含的镜头编号和每个场景在输入视频中的起始帧号和结束帧号
以及对应的起始时间戳和结束时间戳，根据帧号/时间戳可以从原视频中裁剪得到对应的场景段。另外，模型也会给出镜头分割的结果，包括分割得到的镜头数目和每个镜头在输入视频中的起始帧号和结束帧号
以及对应的起始时间戳和结束时间戳。

在ModelScope框架上，提供一段长视频，即可以通过简单的Pipeline调用来使用视频场景分割模型。
如果想要将分割后的场景片段保存在本地，可以将配置文件中`pipeline`字段的`save_split_scene`设置为`true`。

如使用下述范例中默认的输入视频，CPU环境下大约需要8分钟，GPU环境下大约需要1分钟。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

video_scene_seg = pipeline(Tasks.movie_scene_segmentation, model='damo/cv_resnet50-bert_video-scene-segmentation_movienet')
result = video_scene_seg('https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/movie_scene_segmentation_test_video.mp4')
print(result)
```
### 模型局限性和可能的偏差
场景内镜头来回切换过于频繁时可能存在过度分割的现象。

## 训练数据介绍
训练数据为[MovieNet](https://movienet.github.io/)公开数据集。
该数据集中包含318个可用于视频场景分割的电影视频，其中训练集、验证集和测试集分别包含190、64和64个视频。
每个视频按照shot进行了预分割，每个shot提供了三张关键帧图像。

## 模型训练流程
注意：目前本模型只提供微调阶段的训练代码。
本模型在训练的时候使用`Adam`优化器和`CosineAnnealingLR`学习率更新策略。

### 预处理
- 下载[MovieNet](https://movienet.github.io/)数据集提供的关键帧图像压缩包`Movie per-shot keyframes (240P)` 并解压缩；

- 下载BaSSL格式的标注文件（详见本页面模型文件下`data/anno`文件夹），并按照[此格式](https://modelscope.cn/datasets/modelscope/movie_scene_seg_toydata/summary)组织好文件目录；

- 本模型涉及到的预处理主要有以下几个，具体的组织方式见配置文件：
    - `VideoRandomResizedCrop` 随机裁剪
    - `VideoRandomHFlip` 随机翻转
    - `VideoRandomColorJitter` 随机颜色变化
    - `VideoRandomGaussianBlur` 随机高斯模糊
    - `VideoResizedCenterCrop` 中心裁剪

## 数据评估及结果
| name | Dataset | AP | F1 |
|:---- |:----    |:---- |:----|
| BaSSL | MovieNet | 57.40 | 47.02 |

### 模型评估代码
可通过如下代码对模型进行评估验证，目前的评估数据为ModelScope的DataHub上的[toydata](https://modelscope.cn/datasets/modelscope/movie_scene_seg_toydata/summary)，评估完整的MovieNet或者自定义数据请按照toydata对应的方式组织数据和标签文件。
```python
import os
import tempfile
from modelscope.utils.config import Config, ConfigDict
from modelscope.msdatasets import MsDataset
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.constant import ModelFile

model_id = 'damo/cv_resnet50-bert_video-scene-segmentation_movienet'
cache_path = snapshot_download(model_id)
config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)
cfg = Config.from_file(config_path)

test_data_cfg = ConfigDict(
    name='movie_scene_seg_toydata',
    split='test',
    cfg=cfg.preprocessor,
    test_mode=True)

test_dataset = MsDataset.load(
    dataset_name=test_data_cfg.name,
    split=test_data_cfg.split,
    cfg=test_data_cfg.cfg,
    test_mode=test_data_cfg.test_mode)

tmp_dir = tempfile.TemporaryDirectory().name

kwargs = dict(
    model=model_id,
    train_dataset=None,
    eval_dataset=test_dataset,
    work_dir=tmp_dir)

trainer = build_trainer(name=Trainers.movie_scene_segmentation, default_args=kwargs)
metrics = trainer.evaluate()
print(metrics)
```
### 模型训练代码
可通过如下代码对模型进行训练，目前的训练数据为ModelScope的DataHub上的[toydata](https://modelscope.cn/datasets/modelscope/movie_scene_seg_toydata/summary)，训练完整的MovieNet或者自定义数据请按照toydata对应的方式组织数据和标签文件。（微调阶段的训练代码，暂不提供预训练阶段的训练代码）：
```python
import os
import tempfile
from modelscope.utils.config import Config, ConfigDict
from modelscope.msdatasets import MsDataset
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.utils.constant import ModelFile
from modelscope.trainers import build_trainer

model_id = 'damo/cv_resnet50-bert_video-scene-segmentation_movienet'
cache_path = snapshot_download(model_id)
config_path = os.path.join(cache_path, ModelFile.CONFIGURATION)
cfg = Config.from_file(config_path)

train_data_cfg = ConfigDict(
    name='movie_scene_seg_toydata',
    split='train',
    cfg=cfg.preprocessor,
    test_mode=False)

train_dataset = MsDataset.load(
    dataset_name=train_data_cfg.name,
    split=train_data_cfg.split,
    cfg=train_data_cfg.cfg,
    test_mode=train_data_cfg.test_mode)

test_data_cfg = ConfigDict(
    name='movie_scene_seg_toydata',
    split='test',
    cfg=cfg.preprocessor,
    test_mode=True)

test_dataset = MsDataset.load(
    dataset_name=test_data_cfg.name,
    split=test_data_cfg.split,
    cfg=test_data_cfg.cfg,
    test_mode=test_data_cfg.test_mode)

tmp_dir = tempfile.TemporaryDirectory().name


kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    work_dir=tmp_dir)

trainer = build_trainer(name=Trainers.movie_scene_segmentation, default_args=kwargs)
trainer.train()
results_files = os.listdir(trainer.work_dir)
print(results_files)
```

也可通过unittest代码直接调取

```bash
PYTHONPATH=. python tests/run.py --pattern test_movie_scene_segmentation_trainer.py --level 1
```

### 相关论文以及引用信息
如果你觉得这个模型对你有所帮助，请考虑引用下面的相关论文：
```BibTeX
@article{mun2022boundary,
  title={Boundary-aware Self-supervised Learning for Video Scene Segmentation},
  author={Mun, Jonghwan and Shin, Minchul and Han, Gunsoo and Lee, Sangho and Ha, Seongsu and Lee, Joonseok and Kim, Eun-Sol},
  journal={arXiv preprint arXiv:2201.05277},
  year={2022}
}
```
