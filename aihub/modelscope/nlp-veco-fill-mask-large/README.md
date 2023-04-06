
# 基于veco的多语言完形填空模型介绍

基于veco的多语言完形填空模型是用CommonCrawl Corpus 训练的自然语言理解多语言预训练模型。

## 模型描述

VECO是一个联合多语言理解（NLU）和语言生成（NLG）的模型，支持50种语言。通过即插即用的交叉注意力模块，VECO可以更加“显式”得建模语言之间的相互依存关系。基于其灵活的特性，VECO可以同时用于初始化NLU模型的编码器和NLG模型的编码器及解码器。

![veco](./resources/model.png)

## 期望模型使用方式以及适用范围
本模型主要用于多语言完形填空任务。用户可以自行尝试各种输入。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope-lib之后即可使用nlp_veco_fill-mask-large的能力。（注意：模型运行约需占用5.5G内存）

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

fill_mask_multilingual = pipeline(Tasks.fill_mask, model='damo/nlp_veco_fill-mask-large')
result_en = fill_mask_multilingual('Everything in <mask> you call reality is really <mask> a reflection of your <mask>. Your <mask> universe is just a mirror <mask> of your story.')
result_zh = fill_mask_multilingual('段誉轻<mask>折扇，摇了摇<mask>，<mask>道：“你师父是你的<mask><mask>，你师父可不是<mask>的师父。你师父差得动你，你师父可<mask>不动我。')

print(result_en['text'])
print(result_zh['text'])
```

### 模型局限性以及可能的偏差
模型训练数据有限，效果可能存在一定偏差。

## 训练数据介绍

VECO使用50种语言的数据，其中包括基于Common-Crawl Corpus的1.36TB单语数据，包含6.5G句子和0.4G文档，以及基于OPUS的6.4G平行语料，涉及50种语言间的879种语言对。
数据来源于 https://huggingface.co/datasets/cc100  和 http://opus.nlpl.eu/

## 模型训练流程

VECO的模型每一层是一个可变的Transformer层，包括两个可选的Self-attention模块和Cross-attention模块以及一个必选的FFN模块，其Pre-training阶段和Fine-tuning阶段采取“排列组合”和“拆分整合”的方式来训练：在Pre-training阶段通过设计三个不同的任务重新整合三个模块：Self-attention + FFN、Cross-attention + FFN、 Self-attention + Cross-attention + FFN，其中后两个预训练任务相较于TLM（双语数据拼接作为输入的方式执行MLM），可以更显式的利用双语数据。模型在64张Nvidia Telsa V100 32GB GPUs上训练。

### 预处理
暂无
### 训练
```python 
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
import os

langs = ['en']
langs_eval = ['en']
train_datasets = []
for lang in langs:
    train_datasets.append(
        MsDataset.load('xnli', language=lang, split='train'))
eval_datasets = []
for lang in langs_eval:
    eval_datasets.append(
        MsDataset.load(
            'xnli', language=lang, split='validation'))
train_len = sum([len(dataset) for dataset in train_datasets])
labels = [0, 1, 2]


def cfg_modify_fn(cfg):
    cfg.task = 'nli'
    cfg['preprocessor'] = {'type': 'nli-tokenizer'}
    cfg['dataset'] = {
        'train': {
            'first_sequence': 'premise',
            'second_sequence': 'hypothesis',
            'labels': labels,
            'label': 'label',
        }
    }
    cfg['train'] = {
        'work_dir':
        '/tmp',
        'max_epochs':
        2,
        'dataloader': {
            'batch_size_per_gpu': 16,
            'workers_per_gpu': 0
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': 2e-5,
            'options': {
                'cumulative_iters': 8,
            }
        },
        'lr_scheduler': {
            'type': 'LinearLR',
            'start_factor': 1.0,
            'end_factor': 0.0,
            'total_iters': int(train_len / 16) * 2,
            'options': {
                'by_epoch': False
            }
        },
        'hooks': [{
            'type': 'CheckpointHook',
            'interval': 1
        }, {
            'type': 'TextLoggerHook',
            'interval': 1
        }, {
            'type': 'IterTimerHook'
        }, {
            'type': 'EvaluationHook',
            'interval': 1
        }]
    }
    cfg['evaluation'] = {
        'dataloader': {
            'batch_size_per_gpu': 128,
            'workers_per_gpu': 0,
            'shuffle': False
        }
    }
    return cfg


kwargs = dict(
    model='damo/nlp_veco_fill-mask-large',
    train_dataset=train_datasets,
    eval_dataset=eval_datasets,
    work_dir='/tmp',
    cfg_modify_fn=cfg_modify_fn)

os.environ['LOCAL_RANK'] = '0'
trainer = build_trainer(name='nlp-base-trainer', default_args=kwargs)
trainer.train()
```

## 数据评估及结果
VECO在XTREME榜单上排名前列
![veco_xtreme_score](./resources/veco_xtreme.png)


### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用我们的文章：
```BibTex
@inproceedings{luo-etal-2021-veco,
    title = "{VECO}: Variable and Flexible Cross-lingual Pre-training for Language Understanding and Generation",
    author = "Luo, Fuli  and
      Wang, Wei  and
      Liu, Jiahao  and
      Liu, Yijia  and
      Bi, Bin  and
      Huang, Songfang  and
      Huang, Fei  and
      Si, Luo",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.308",
    doi = "10.18653/v1/2021.acl-long.308",
    pages = "3980--3994",
}
```
