
# 跨模态检索介绍
跨模态检索：给定一个文本和图片，给出图文相似度分数。需要模型具备多模态理解的能力，目前主流的方法大多是基于多模态预训练，最为知名的跨模态检索数据集包括：Flickr30K,COCO等。

## 模型描述

本任务是mPLUG，在英文Flickr30k数据集进行finetune的跨模态检索下游任务。mPLUG模型是统一理解和生成的多模态基础模型，该模型提出了基于skip-connections的高效跨模态融合框架。其中，mPLUG在Flickr30k上取得SOTA，详见：[mPLUG: Effective and Efficient Vision-Language Learning by Cross-modal Skip-connections](https://arxiv.org/abs/2205.12005)

![mplug](./resources/model.png)


## 期望模型使用方式以及适用范围
本模型主要用于给文本和图片生成相似度分数。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成MaaS-lib之后即可使用cross-modal-retrieval的能力 （注意：模型运行约需占用 9G 内存）

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model = 'damo/mplug_image-text-retrieval_flickr30k_large_en'
pipeline_retrieval = pipeline(Tasks.image_text_retrieval, model=model)
image = 'https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/image-text-retrieval.jpg'
question = 'Two young guys with shaggy hair look at their hands while hanging out in the yard.'
input = {'image': image, 'question': question}
result = pipeline_retrieval(input)
print(result)
```

### 模型局限性以及可能的偏差
模型在数据集上训练，有可能产生一些偏差，请用户自行评测后决定如何使用。

## 训练数据介绍
本模型训练数据集是Flickr30k，数据集包含30k图片， 具体数据可以[下载](https://shannon.cs.illinois.edu/DenotationGraph/)

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
    model='damo/mplug_image-text-retrieval_flickr30k_large_en',
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
mPLUG在Flickr30k数据集，同等规模和预训练数据的模型中取得SOTA

![mplug_retrieval_score](./resources/retrieval_exp.png)
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
