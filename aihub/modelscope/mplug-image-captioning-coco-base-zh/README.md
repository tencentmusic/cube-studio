
# 图像描述介绍
图像描述：给定一张图片，模型根据图片信息生成一句对应描述。可以应用于给一张图片配上一句文字或者打个标签的场景。本页面右侧提供了在线体验的服务，欢迎使用！注：本模型为mPLUG-图像描述的中文Base模型，参数量约为3.5亿。

## 模型描述

本任务是mPLUG，在翻译成中文的图像描述MS COCO Caption数据集进行finetune的图像描述下游任务。mPLUG模型是统一理解和生成的多模态基础模型，该模型提出了基于skip-connections的高效跨模态融合框架。其中，mPLUG论文公开时在MS COCO Caption数据上达到SOTA，详见：[mPLUG: Effective and Efficient Vision-Language Learning by Cross-modal Skip-connections](https://arxiv.org/abs/2205.12005)

![mplug](./resources/model.png)


## 期望模型使用方式以及适用范围
本模型主要用于给问题和对应图片生成答案。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成MaaS-lib之后即可使用image-captioning的能力

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/mplug_image-captioning_coco_base_zh'
input_caption = 'https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/image_captioning.png'

pipeline_caption = pipeline(Tasks.image_captioning, model=model_id)
result = pipeline_caption(input_caption)
print(result)

```

### 模型局限性以及可能的偏差
模型在数据集上训练，有可能产生一些偏差，请用户自行评测后决定如何使用。

## 训练数据介绍
本模型训练数据集是MS COCO Caption， 具体数据可以[下载](https://cocodataset.org)

## 模型训练流程

### 微调代码范例

```python
import tempfile

from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

datadict = MsDataset.load('coco_captions_small_slice')

train_dataset = MsDataset(
    datadict['train'].remap_columns({
        'image:FILE': 'image',
        'answer:Value': 'answer'
    }).map(lambda _: {'question': 'what the picture describes?'}))
test_dataset = MsDataset(
    datadict['test'].remap_columns({
        'image:FILE': 'image',
        'answer:Value': 'answer'
    }).map(lambda _: {'question': 'what the picture describes?'}))

# 可以在代码修改 configuration 的配置
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
    return cfg

kwargs = dict(
    model='damo/mplug_image-captioning_coco_base_zh',
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    max_epochs=2,
    cfg_modify_fn=cfg_modify_fn,
    work_dir=tempfile.TemporaryDirectory().name)

trainer = build_trainer(
    name=Trainers.nlp_base_trainer, default_args=kwargs)
trainer.train()
```

## 数据评估及结果
mPLUG在VQA数据集，同等规模和预训练数据的模型中取得SOTA，VQA榜单上排名前列

![mplug_caption_score](./resources/caption_exp.png)

### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引入我们的文章：
```BibTeX
@inproceedings{li2022mplug,
      title={mPLUG: Effective and Efficient Vision-Language Learning by Cross-modal Skip-connections}, 
      author={Li, Chenliang and Xu, Haiyang and Tian, Junfeng and Wang, Wei and Yan, Ming and Bi, Bin and Ye, Jiabo and Chen, Hehong and Xu, Guohai and Cao, Zheng and Zhang, Ji and Huang, Songfang and Huang, Fei and Zhou, Jingren and Luo Si},
      year={2022},
      journal={arXiv}
}
```



