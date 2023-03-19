## News
- 2023年1月: 优化了finetune流程，支持参数更新、自定义数据及脚本分布式训练等，见finetune示例。
- 2022年09月: 上线[Huge模型](https://modelscope.cn/models/damo/ofa_image-caption_coco_huge_en/summary)，欢迎试用。

# OFA-图像描述(英文)


## 图像描述是什么？
如果你希望为一张图片配上一句文字，或者打个标签，OFA模型就是你的绝佳选择。你只需要输入任意1张你的图片，**3秒内**就能收获一段精准的描述。**本页面右侧**提供了在线体验的服务，欢迎使用！

本系列还有如下模型，欢迎试用：
- [large-通用场景-英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_large_en/summary)
- [huge-通用场景-英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_huge_en/summary)
- [base-电商场景-中文](https://modelscope.cn/models/damo/ofa_image-caption_muge_base_zh/summary)

## 快速玩起来
玩转OFA只需区区以下6行代码，就是如此轻松！如果你觉得还不够方便，请点击右上角`Notebook`按钮，我们为你提供了配备了GPU的环境，你只需要在notebook里输入提供的代码，就可以把OFA玩起来了！

<p align="center">
    <img src="resources/donuts.jpeg" alt="donuts" width="200" />

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

img_captioning = pipeline(Tasks.image_captioning, model='damo/ofa_image-caption_coco_distilled_en', model_revision='v1.0.1')
result = img_captioning('https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg')
print(result[OutputKeys.CAPTION])  # 'a wooden table topped with different types of donuts'
```
<br>

## OFA是什么？
OFA(One-For-All)是通用多模态预训练模型，使用简单的序列到序列的学习框架统一模态（跨模态、视觉、语言等模态）和任务（如图片生成、视觉定位、图片描述、图片分类、文本生成等），详见我们发表于ICML 2022的论文：[OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://arxiv.org/abs/2202.03052)，以及我们的官方Github仓库[https://github.com/OFA-Sys/OFA](https://github.com/OFA-Sys/OFA)。

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
        <video src="https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/resources/modelscope_web/demo.mp4" loop="loop" autoplay="autoplay" muted width="100%"></video>
    <br>
</p>

## OFA的轻量化

本模型是在对OFA的large模型版本基础上通过知识蒸馏进行轻量化而得到的tiny版本模型（参数量33M），方便用户部署在存储和计算资源受限的设备上。蒸馏框架介绍，详见论文：[Knowledge Distillation of Transformer-based Language Models Revisited](https://arxiv.org/abs/2206.14366)，以及我们的官方Github仓库[https://github.com/OFA-Sys/OFA-Compress](https://github.com/OFA-Sys/OFA-Compress)

![distill](./resources/distill_framework.png)

模型效果如下：

![ofa-image-caption](./resources/caption_demo.png)

<table>
<thead>
  <tr>
    <th></th>
    <th>OFA-tiny<br>(直接finetune得到)</th>
    <th>OFA-distill-tiny<br>(通过蒸馏得到)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CIDEr</td>
    <td>119.0</td>
    <td>120.1</td>
  </tr>
</tbody>
</table>

## 模型训练流程

### 训练数据介绍
本模型训练数据集是coco caption。

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
    cfg.train.max_epochs=2
    return cfg


args = dict(
    model='damo/ofa_image-caption_coco_distilled_en', 
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

## 相关论文以及引用信息
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：
```
@article{Lu2022KnowledgeDO,
  author    = {Chengqiang Lu and 
               Jianwei Zhang and 
               Yunfei Chu and 
               Zhengyu Chen and 
               Jingren Zhou and 
               Fei Wu and 
               Haiqing Chen and 
               Hongxia Yang},
  title     = {Knowledge Distillation of Transformer-based Language Models Revisited},
  journal   = {ArXiv},
  volume    = {abs/2206.14366}
  year      = {2022}
}
```
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