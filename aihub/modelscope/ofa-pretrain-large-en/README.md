# OFA预训练模型
## News
- 2023年1月：
  - 优化了finetune流程，支持参数更新、自定义数据及脚本分布式训练等，见finetune示例。
- 2022年11月：
  - 发布ModelScope 1.0版本，以下能力请使用1.0及以上版本。
  - 新增[OFA Tutorial](https://test.modelscope.cn/docs/OFA%20Tutorial)。

## 如何使用OFA预训练模型

### Finetune
OFA预训练模型是OFA在8个预训练任务（具体参见论文）上得到ckpt，是finetune下游任务的基础。

模型细节参考文档：[OFA Tutorial](https://modelscope.cn/docs/OFA_Tutorial#1.4%20%E5%A6%82%E4%BD%95%E8%AE%AD%E7%BB%83) 1.4节。

这里直接使用预训练模型在caption任务上进行实例演示，要求 ModelScope Library >= 1.2.0

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

pretrained_model = 'damo/ofa_pretrain_large_en'
pretrain_path = snapshot_download(pretrained_model, revision='v1.0.2')

args = dict(
    model=pretrain_path,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    cfg_modify_fn=cfg_modify_fn,
    work_dir = tempfile.TemporaryDirectory().name)
trainer = build_trainer(name=Trainers.ofa, default_args=args)
trainer.train()
```
### ZeroShot
OFA预训练模型可以通过以下代码进行推理测试

有了预训练模型，可以OFA自身的能力特点（One For All）利用预训练模型测试其在下游任务的效果，具体以caption能力作为示例：

```python
import os
import shutil
from modelscope.utils.hub import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile

pretrained_model = 'damo/ofa_pretrain_large_en'
pretrain_path = snapshot_download(pretrained_model, revision='v1.0.2')
task_model = 'damo/ofa_image-caption_coco_large_en'
task_path = snapshot_download(task_model)

shutil.copy(os.path.join(task_path, ModelFile.CONFIGURATION), # 将任务的配置覆盖预训练模型的配置
            os.path.join(pretrain_path, ModelFile.CONFIGURATION))

ofa_pipe = pipeline(Tasks.image_captioning, model=pretrain_path)
result = ofa_pipe('https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg')
print(result[OutputKeys.CAPTION]) 
```

# OFA是什么？
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

# OFA模型规模

| **Model** | **Params-en** | **Params-zh** | **Backbone** | **Hidden size** | **Intermediate size** | **Num. of heads** | **Enc layers** | **Dec layers** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OFA<sub>Tiny</sub> | 33M | - | ResNet50 | 256 | 1024 | 4 | 4 | 4 |
| OFA<sub>Medium</sub> | 93M | - | ResNet101 | 512 | 2048 | 8 | 4 | 4 |
| OFA<sub>Base</sub> | 180M | 160M | ResNet101 | 768 | 3072 | 12 | 6 | 6 |
| OFA<sub>Large</sub> | 470M | 443M | ResNet152 | 1024 | 4096 | 16 | 12 | 12 |
| OFA<sub>Huge</sub> | 930M | - | ResNet152 | 1280 | 5120 | 16 | 24 | 12 |
| OFA<sub>6B</sub> | 6B | - | vit_huge | 2560 | 10240 | 32 | 36 | 24 |

<br>

# OFA 模型任务矩阵
目前ModelScope上面所有已经上传的模型和任务可以在下面导航表格看到，点击链接可以跳转到相应modelcard。


| 模型规模 | 预训练 | 图像描述 | 视觉问答 | 视觉定位 | 视觉蕴含 | 文生图 | 图像分类 | 文字识别 | 文本摘要 | 文本分类 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OFA<sub>Tiny</sub> | [英文](https://modelscope.cn/models/damo/ofa_pretrain_tiny_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_distilled_en/summary) | - | [英文](https://modelscope.cn/models/damo/ofa_visual-grounding_refcoco_distilled_en) | [英文](https://modelscope.cn/models/damo/ofa_visual-entailment_snli-ve_distilled_v2_en/summary) | - | - | - | - | - |
| OFA<sub>Medium</sub> | [英文](https://modelscope.cn/models/damo/ofa_pretrain_medium_en/summary)  | - | - | - | - | - | - | - | - | - |
| OFA<sub>Base</sub> | [中文](https://modelscope.cn/models/damo/ofa_pretrain_base_zh/summary)/[英文](https://modelscope.cn/models/damo/ofa_pretrain_base_en/summary) | [中文电商](https://modelscope.cn/models/damo/ofa_image-caption_muge_base_zh/summary) | - | - | - | - | - | [场景中文](https://modelscope.cn/models/damo/ofa_ocr-recognition_scene_base_zh/summary) | - | - |
| OFA<sub>Large</sub> | [中文](https://modelscope.cn/models/damo/ofa_pretrain_large_zh/summary)/[英文](https://modelscope.cn/models/damo/ofa_pretrain_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_visual-question-answering_pretrain_large_en/summary) | [中文](https://modelscope.cn/models/damo/ofa_visual-grounding_refcoco_large_zh/summary)/[英文](https://modelscope.cn/models/damo/ofa_visual-grounding_refcoco_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_visual-entailment_snli-ve_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_text-to-image-synthesis_coco_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_image-classification_imagenet_large_en/summary) | - | [英文](https://modelscope.cn/models/damo/ofa_summarization_gigaword_large_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_text-classification_mnli_large_en/summary) |
| OFA<sub>Huge</sub> | [英文](https://modelscope.cn/models/damo/ofa_pretrain_huge_en/summary)  | [英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_huge_en/summary) | [英文](https://modelscope.cn/models/damo/ofa_visual-question-answering_pretrain_huge_en/summary) | - | - | - | - | - | - | - |
| OFA<sub>6B</sub> | - | [英文](https://modelscope.cn/models/damo/ofa_image-caption_coco_6b_en/summary) | - | - | - | - | - | - | - | - |


# 相关论文以及引用
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