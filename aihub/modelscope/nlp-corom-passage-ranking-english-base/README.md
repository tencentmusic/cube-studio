
# CoROM语义相关性-英文-通用领域模型介绍

文本检索是信息检索领域的核心问题, 其在很多信息检索、NLP下游任务中发挥着非常重要的作用。 近几年, BERT等大规模预训练语言模型的出现使得文本表示效果有了大幅度的提升, 基于预训练语言模型构建的文本检索系统在召回、排序效果上都明显优于传统统计模型。

由于文档候选集合通常比较庞大，实际的工业搜索系统中候选文档数量往往在千万甚至更高的数量级, 为了兼顾效率和准确率，目前的文本检索系统通常是基于召回&排序的多阶段搜索框架。在召回阶段，系统的主要目标是从海量文本中去找到潜在跟query相关的文档，得到较小的候选文档集合（100-1000个）。召回完成后, 排序阶段的模型会对这些召回的候选文档进行更加复杂的排序, 产出最后的排序结果。本模型为基于预训练的排序阶段模型。

## 模型描述
本模型为基于CoROM-Base预训练模型的通用领域英文语义相关性模型，模型以一个source sentence以及一个句子列表作为输入，最终输出source sentence与列表中每个句子的相关性得分（0-1，分数越高代表两者越相关）

<div align=center><img height="300" src="./resource/reranker.png" /></div>

### 期望模型使用方式以及适用范围
本模型主要用于给输入英文查询与文档列表产出相关性分数。用户可以自行尝试输入查询和文档。具体调用方式请参考代码示例。

### 如何使用
在安装ModelScope完成之后即可使用语义相关性模型, 该模型以一个source sentence以及一个“sentence_to_compare"（句子列表）作为输入，最终输出source sentence与列表中每个句子的相关性得分（0-1，分数越高代表两者越相关）。 默认每个句子对长度不超过512。

#### 代码范例

```python
# 可在CPU/GPU环境运行
from modelscope.models import Model
from modelscope.pipelines import pipeline
# Version less than 1.1 please use TextRankingPreprocessor
from modelscope.preprocessors import TextRankingTransformersPreprocessor
from modelscope.utils.constant import Tasks

input = { 
    'source_sentence': ["how long it take to get a master's degree"],
    'sentences_to_compare': [
        "On average, students take about 18 to 24 months to complete a master's degree.",
        'On the other hand, some students prefer to go at a slower pace and choose to take '
        'several years to complete their studies.',
        'It can take anywhere from two semesters'
    ]   
}
model_id = 'damo/nlp_corom_passage-ranking_english-base'
model = Model.from_pretrained(model_id)
preprocessor = TextRankingTransformersPreprocessor(model.model_dir)
pipeline_ins = pipeline(task=Tasks.text_ranking, model=model, preprocessor=preprocessor)
result = pipeline_ins(input=input)
print (result)
# {'scores': [0.9292812943458557, 0.2204243242740631, 0.4248475730419159]}
```

### 模型局限性以及可能的偏差
本模型基于MS MARCO Passage数据集(公开篇章排序数据集)上训练，在垂类领域上的排序效果会有降低，请用户自行评测后决定如何使用

## 模型训练

### 训练流程
- 模型: 单塔篇章排序模型, 采用coROM模型作为预训练语言模型底座
- 训练数据: 本模型采用英文篇章排序数据集MS MARCO Passage标注训练,训练过程中，正样本来自官方标注数据，负样本来自coCondenser模型召回负样本。


模型采用4张NVIDIA V100机器训练, 主要超参设置如下: 
```
train_epochs=3
max_sequence_length=128                                                                                                                                                      
batch_size=128
learning_rate=1e-5
optimizer=AdamW                                                                                                                                                              
neg_samples=8
```
### 训练示例代码

```python
# 需在GPU环境运行
# 加载数据集过程可能由于网络原因失败，请尝试重新运行代码
from modelscope.metainfo import Trainers                                                                                                                                                              
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
import tempfile
import os

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# load dataset
ds = MsDataset.load('msmarco-passage-ranking', 'zyznull')
train_ds = ds['train'].to_hf_dataset()
dev_ds = ds['dev'].to_hf_dataset()
model_id = 'damo/nlp_corom_passage-ranking_english-base'
def cfg_modify_fn(cfg):
    neg_sample = 4
    cfg.task = 'text-ranking'
    cfg['preprocessor'] = {'type': 'text-ranking'}
    cfg.train.optimizer.lr = 2e-5
    cfg['dataset'] = {
        'train': {
            'type': 'bert',
            'query_sequence': 'query',
            'pos_sequence': 'positive_passages',
            'neg_sequence': 'negative_passages',
            'text_fileds': ['title', 'text'],
            'qid_field': 'query_id',
            'neg_sample': neg_sample
        },
        'val': {
            'type': 'bert',
            'query_sequence': 'query',
            'pos_sequence': 'positive_passages',
            'neg_sequence': 'negative_passages',
            'text_fileds': ['title', 'text'],
            'qid_field': 'query_id'
        },
    }
    cfg['evaluation']['dataloader']['batch_size_per_gpu'] = 30
    cfg.train.max_epochs = 1
    cfg.train.train_batch_size = 4
    cfg.train.lr_scheduler = {
        'type': 'LinearLR',
        'start_factor': 1.0,
        'end_factor': 0.0,
        'options': {
            'by_epoch': False
        }
    }
    cfg.model['neg_sample'] = 4
    cfg.train.hooks = [{
        'type': 'CheckpointHook',
        'interval': 1
    }, {
        'type': 'TextLoggerHook',
        'interval': 1
    }, {
        'type': 'IterTimerHook'
    }, {
        'type': 'EvaluationHook',
        'by_epoch': False,
        'interval': 5000
    }]
    return cfg
kwargs = dict(
    model=model_id,
    train_dataset=train_ds,
    work_dir=tmp_dir,
    eval_dataset=dev_ds,
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(name=Trainers.nlp_text_ranking_trainer, default_args=kwargs)
trainer.train()
```

## 数据评估及结果
本模型在MS MARCO Passage Ranking 利用[CoCondenser](https://huggingface.co/Luyu/co-condenser-marco-retriever)召回top100上的排序结果如下:

| Model     | MRR@10 |
|-----------|--------|
| BERT      |  40.1  |
| RoBERTa    |  40.8  |
| CoROM     |  41.3  |


### 模型评估代码
```python
# 需在GPU环境运行
# 加载数据集过程可能由于网络原因失败，请尝试重新运行代码
from modelscope.metainfo import Trainers                                                                                                                                                              
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
import tempfile
import os

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# load dataset
ds = MsDataset.load('msmarco-passage-ranking', 'zyznull')
dev_ds = ds['dev'].to_hf_dataset()
model_id = 'damo/nlp_corom_passage-ranking_english-base'
def cfg_modify_fn(cfg):
    cfg.task = 'text-ranking'
    cfg['preprocessor'] = {'type': 'text-ranking'}
    cfg['dataset'] = { 
        'val': {
            'type': 'bert',
            'query_sequence': 'query',
            'pos_sequence': 'positive_passages',
            'neg_sequence': 'negative_passages',
            'passage_text_fileds': ['title','text'],
            'qid_field': 'query_id'
        },
    }   
    cfg['evaluation']['dataloader']['batch_size_per_gpu'] = 32
    return cfg 
kwargs = dict(
    model=model_id,
    train_dataset=None,
    work_dir=tmp_dir,
    eval_dataset=dev_ds,
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(name=Trainers.nlp_text_ranking_trainer, default_args=kwargs)
# evalute 在单GPU（V100）上需要约十分钟，请耐心等待
metric_values = trainer.evaluate()
print(metric_values)
```

## 引用
预训练语言模型应用于文本相关性、文本检索排序可以参考论文
```BibTex
@article{zhangHLATR,
  author    = {Yanzhao Zhang and Dingkun Long and Guangwei Xu and Pengjun Xie},
  title     = {{HLATR:} Enhance Multi-stage Text Retrieval with Hybrid List Aware Transformer Reranking},
  journal   = {CoRR},
  volume    = {abs/2205.10569},
  year      = {2022}
}
```