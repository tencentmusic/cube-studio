# OFA-图像描述(英文)
## News
- 2023年1月:
  - 优化了finetune流程，支持参数更新、自定义数据及脚本分布式训练等，见finetune示例。
- 2022年12月：
  - 支持了batch inference，具体见本页`快速玩起来`demo示例
- 2022年11月：
  - 发布ModelScope 1.0版本，以下能力请使用1.0.2及以上版本。
  - 上线[6B Caption模型](https://modelscope.cn/models/damo/ofa_image-caption_coco_6b_en/summary)
  - 支持finetune能力，新增[OFA Tutorial](https://modelscope.cn/docs/OFA_Tutorial#1.4%20%E5%A6%82%E4%BD%95%E8%AE%AD%E7%BB%83)，finetune能力参考1.4节。


## 图像描述是什么？
如果你希望为一张图片配上一句文字，或者打个标签，OFA模型就是你的绝佳选择。你只需要输入任意1张你的图片，**3秒内**就能收获一段精准的描述。**本页面右侧**提供了在线体验的服务，欢迎使用！

本系列还有如下模型，欢迎试用：
- [tiny-通用场景-英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_distilled_en/summary)
- [large-通用场景-英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_large_en/summary)
- [base-电商场景-中文](https://modelscope.cn/models/damo/ofa_image-caption_muge_base_zh/summary)

## 快速玩起来
玩转OFA只需区区以下6行代码，就是如此轻松！如果你觉得还不够方便，请点击右上角Notebook按钮，我们为你提供了配备了的环境，CPU和GPU都支持哦，你只需要在notebook里输入提供的代码，就可以把OFA玩起来了!

<p align="center">
    <img src="resources/donuts.jpg" alt="donuts" width="200" />

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

img_captioning = pipeline(Tasks.image_captioning, model='damo/ofa_image-caption_coco_huge_en', model_revision='v1.0.1')
result = img_captioning('https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg')
print(result[OutputKeys.CAPTION]) # 'a bunch of donuts on a wooden board with popsicle sticks'
# 目前caption支持了batch inference，方式非常简单，具体如下：
result = img_captioning([{'image': 'https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg'} for _ in range(3)], batch_size=2)
for r in result:
    print(r[OutputKeys.CAPTION])
```


## OFA是什么？
OFA(One-For-All)是通用多模态预训练模型，使用简单的序列到序列的学习框架统一模态（跨模态、视觉、语言等模态）和任务（如图片生成、视觉定位、图片描述、图片分类、文本生成等），详见我们发表于ICML 2022的论文：[OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052)，以及我们的官方Github仓库[https://github.com/OFA-Sys/OFA](https://github.com/OFA-Sys/OFA)。
<br><br>

<p align="center">
    <br>
    <img src="resources/OFA_logo_tp_path.svg" width="150" />
    <br>
<p>
<br>

<p align="center">
        <a href="https://github.com/OFA-Sys/OFA">Github</a>&nbsp ｜ &nbsp<a href="https://arxiv.org/abs/2202.03052">Paper </a>&nbsp ｜ &nbspBlog
</p>

<p align="center">
    <br>
        <video src="https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/resources/modelscope_web/demo.mp4" loop="loop" autoplay="autoplay" muted width="80%"></video>
    <br>
</p>

### OFA模型规模：

<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>Params-en</th><th>Params-zh</th><th>Backbone</th><th>Hidden size</th><th>Intermediate size</th><th>Num. of heads</th><th>Enc layers</th><th>Dec layers</th>
    </tr>
    <tr align="center">
        <td>OFA<sub>Tiny</sub></td><td>33M</td><td>-</td><td>ResNet50</td><td>256</td><td>1024</td><td>4</td><td>4</td><td>4</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Medium</sub></td><td>93M</td><td>-</td><td>ResNet101</td><td>512</td></td><td>2048</td><td>8</td><td>4</td><td>4</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Base</sub></td><td>180M</td><td>160M</td><td>ResNet101</td><td>768</td></td><td>3072</td><td>12</td><td>6</td><td>6</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Large</sub></td><td>470M</td><td>440M</td><td>ResNet152</td><td>1024</td></td><td>4096</td><td>16</td><td>12</td><td>12</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Huge</sub></td><td>930M</td><td>-</td><td>ResNet152</td><td>1280</td></td><td>5120</td><td>16</td><td>24</td><td>12</td>
    </tr>
</table>
<br>


## 为什么OFA是图像描述的最佳选择？
OFA在图像描述（image captioning）任务的权威数据集Microsoft COCO Captions官方榜单成功登顶（想看榜单，点[这里](https://competitions.codalab.org/competitions/3221#results)），并在经典测试集Karpathy test split取得CIDEr 154.9的分数。具体如下：

<table border="1" width="100%">
    <tr align="center">
        <th>Stage</th><th colspan="4">Cross Entropy Optimization</th><th colspan="4">CIDEr Optimization</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>BLEU-4</td><td>METEOR</td><td>CIDEr</td><td>SPICE</td><td>BLEU-4</td><td>METEOR</td><td>CIDEr</td><td>SPICE</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Base</sub></td><td>41.0</td><td>30.9</td><td>138.2</td><td>24.2</td><td>42.8</td><td>31.7</td><td>146.7</td><td>25.8</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Large</sub></td><td>42.4</td><td>31.5</td><td>142.2</td><td>24.5</td><td>43.6</td><td>32.2</td><td>150.7</td><td>26.2</td>
    </tr>
    <tr align="center">
        <td><b>OFA<sub>Huge</sub></b></td><td>43.9</td><td>31.8</td><td>145.3</td><td>24.8</td><td>44.9</td><td>32.5</td><td>154.9</td><td>26.6</td>
    </tr>
</table>
<br>

## 模型训练流程

### 训练数据介绍
本模型训练数据集是MS COCO Caption。

### 训练流程
模型及finetune细节请参考[OFA Tutorial](https://modelscope.cn/docs/OFA_Tutorial#1.4%20%E5%A6%82%E4%BD%95%E8%AE%AD%E7%BB%83) 1.4节。

### Finetune示例
```python
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.hub import snapshot_download


train_dataset = MsDataset(
    MsDataset.load(
        "coco_2014_caption", 
        namespace="modelscope", 
        split="train[:100]",
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS).remap_columns({
        'image': 'image',
        'caption': 'text'
    }))
test_dataset = MsDataset(
    MsDataset.load(
        "coco_2014_caption", 
        namespace="modelscope", 
        split="validation[:20]",
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS).remap_columns({
        'image': 'image',
        'caption': 'text'
    }))


def cfg_modify_fn(cfg):
    cfg.train.hooks = [{
        'type': 'CheckpointHook',
        'interval': 2
    }, {
        'type': 'TextLoggerHook',
        'interval': 1
    }, {
        'type': 'IterTimerHook'
    }]
    cfg.train.dataloader.batch_size_per_gpu = 1
    cfg.train.max_epochs=2
    return cfg


args = dict(
    model='damo/ofa_image-caption_coco_huge_en',
    model_revision='v1.0.1',
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    cfg_modify_fn=cfg_modify_fn,
    work_dir = tempfile.TemporaryDirectory().name)
trainer = build_trainer(name=Trainers.ofa, default_args=args)
trainer.train()
```

## 模型局限性以及可能的偏差
训练数据集自身有局限，有可能产生一些偏差，请用户自行评测后决定如何使用。
## 相关论文以及引用
如果你觉得OFA好用，喜欢我们的工作，欢迎引用：
```
@article{wang2022ofa,
  author    = {Peng Wang and
               An Yang and
               Rui Men and
               Junyang Lin and
               Shuai Bai and
               Zhikang Li and
               Jianxin Ma and
               Chang Zhou and
               Jingren Zhou and
               Hongxia Yang},
  title     = {OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence
               Learning Framework},
  journal   = {CoRR},
  volume    = {abs/2202.03052},
  year      = {2022}
}
```