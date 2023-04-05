
# coROM英文通用文本表示模型

文本表示是自然语言处理(NLP)领域的核心问题, 其在很多NLP、信息检索的下游任务中发挥着非常重要的作用。近几年, 随着深度学习的发展，尤其是预训练语言模型的出现极大的推动了文本表示技术的效果, 基于预训练语言模型的文本表示模型在学术研究数据、工业实际应用中都明显优于传统的基于统计模型或者浅层神经网络的文本表示模型。这里, 我们主要关注基于预训练语言模型的文本表示。

文本表示示例, 输入一个句子, 输入一个固定维度的连续向量:

- 输入: how long it take to get a master\'s degree
- 输出: [0.27162,-0.66159,0.33031,0.24121,0.46122,...]

文本的向量表示通常可以用于文本聚类、文本相似度计算、文本向量召回等下游任务中。

## Dual Encoder文本表示模型

基于监督数据训练的文本表示模型通常采用Dual Encoder框架, 如下图所示。在Dual Encoder框架中, Query和Document文本通过预训练语言模型编码后, 通常采用预训练语言模型[CLS]位置的向量作为最终的文本向量表示。基于标注数据的标签, 通过计算query-document之间的cosine或者L2距离度量两者之间的相关性。

<div align=center><img width="450" height="300" src="./resources/dual-encoder.png" /></div>

## 使用方式和范围

使用方式:
- 直接推理, 对给定文本计算其对应的文本向量表示，向量维度768

使用范围:
- 本模型可以使用在通用领域的文本向量表示及其下游应用场景, 包括双句文本相似度计算、query&多doc候选的相似度排序

### 如何使用

在ModelScope框架上，提供输入文本(默认最长文本长度为128)，即可以通过简单的Pipeline调用来使用coROM文本向量表示模型。ModelScope封装了统一的接口对外提供单句向量表示、双句文本相似度、多候选相似度计算功能

#### 代码示例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = "damo/nlp_corom_sentence-embedding_english-tiny"
pipeline_se = pipeline(Tasks.sentence_embedding, 
                       model=model_id)

inputs = {
        "source_sentence": ["how long it take to get a master degree"],
        "sentences_to_compare": [
            "On average, students take about 18 to 24 months to complete a master degree.",
            "On the other hand, some students prefer to go at a slower pace and choose to take",
            "several years to complete their studies.",
            "It can take anywhere from two semesters"
        ]
    }


result = pipeline_se(input=inputs)
print (result)

# {'text_embedding': array([[ 0.08452811, -0.0636334 ,  0.5774376 , ...,  0.1499477 ,
#          0.07685442,  0.06914043],
#        [-0.12009228,  0.1660448 ,  0.35046986, ...,  0.0850232 ,
#         -0.01834037,  0.10846637],
#        [-0.14880717, -0.3792838 , -0.34287834, ...,  0.33967134,
#         -0.12936975, -0.2094945 ],
#        [ 0.37085992,  0.52807516,  0.170942  , ...,  0.00421665,
#          0.00313525, -0.25771397],
#        [ 0.27699593, -0.08881918, -0.08759344, ...,  0.26941332,
#         0.09722027,  0.06628524]], dtype=float32), 'scores': [162.09716796875, 118.86981964111328, 138.30409240722656, 136.58656311035156]}
```
**默认向量维度768, scores中的score计算两个向量之间的L2距离得到**

### 模型局限性以及可能的偏差

本模型基于MS MARCO Passage Ranking英文数据集(通用领域)上训练，在垂类领域英文文本上的文本效果会有降低，请用户自行评测后决定如何使用

## 训练数据介绍

训练数据采用[MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking)官方开源数据

### 训练流程

- 模型: 双塔文本表示模型, 采用coROM模型作为预训练语言模型底座
- 二阶段训练: 模型训练分为两阶段, 一阶段的负样本数据从官方提供的BM25召回数据中采样, 二阶段通过Dense Retrieval挖掘难负样本扩充训练训练数据重新训练

模型采用4张NVIDIA V100机器训练, 超参设置如下:
```
train_epochs=3
max_sequence_length=128
batch_size=64
learning_rate=5e-6
optimizer=AdamW
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
model_id = 'damo/nlp_corom_sentence-embedding_english-tiny'
def cfg_modify_fn(cfg):
    cfg.task = 'sentence-embedding'
    cfg['preprocessor'] = {'type': 'sentence-embedding','max_length': 256}
    cfg['dataset'] = {
        'train': {
            'type': 'bert',
            'query_sequence': 'query',
            'pos_sequence': 'positive_passages',
            'neg_sequence': 'negative_passages',
            'text_fileds': ['text'],
            'qid_field': 'query_id'
        },
        'val': {
            'type': 'bert',
            'query_sequence': 'query',
            'pos_sequence': 'positive_passages',
            'neg_sequence': 'negative_passages',
            'text_fileds': ['text'],
            'qid_field': 'query_id'
        },
    }
    cfg['train']['neg_samples'] = 4
    cfg['evaluation']['dataloader']['batch_size_per_gpu'] = 30
    cfg.train.max_epochs = 1
    cfg.train.train_batch_size = 4
    return cfg 
kwargs = dict(
    model=model_id,
    train_dataset=train_ds,
    work_dir=tmp_dir,
    eval_dataset=dev_ds,
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(name=Trainers.nlp_sentence_embedding_trainer, default_args=kwargs)
trainer.train()
```

### 模型效果评估

我们主要在文本向量召回场景下评估模型效果, MS MARCO Passage Ranking Dev评估结果:

| Model       | MRR@10 | Recall@1000 |
|-------------|--------|-------------|
| BERT-base   | 33.4   | 95.5        |
| ANCE        | 33.0   | 95.5        |
| ME-BERT     | 33.8   | -           |
| RecketQA    | 37.0   | 97.9        |
| Condenser   | 36.6   | 97.4        |
| coCondenser | 38.2   | 98.4        |
| coROM-base       | 39.1   | 98.6        |
| coROM-tiny       | 37.3   | 97.4        |

## 引用

```BibTex
@article{msmarcoData,
  author    = {Tri Nguyen and Mir Rosenberg and Xia Song and Jianfeng Gao and Saurabh Tiwary and Rangan Majumder and Li Deng},
  title     = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  journal   = {CoRR},
  volume    = {abs/1611.09268},
  year      = {2016}
}
```