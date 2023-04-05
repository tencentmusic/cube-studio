
# coROM中文电商文本表示模型

文本表示是自然语言处理(NLP)领域的核心问题, 其在很多NLP、信息检索的下游任务中发挥着非常重要的作用。近几年, 随着深度学习的发展，尤其是预训练语言模型的出现极大的推动了文本表示技术的效果, 基于预训练语言模型的文本表示模型在学术研究数据、工业实际应用中都明显优于传统的基于统计模型或者浅层神经网络的文本表示模型。这里, 我们主要关注基于预训练语言模型的文本表示。

文本表示示例, 输入一个句子, 输入一个固定维度的连续向量:

- 输入: 阔腿裤女冬牛仔
- 输出: [-0.23219466,  0.41309455,  0.26903808, ..., -0.276916]

文本的向量表示通常可以用于文本聚类、文本相似度计算、文本向量召回等下游任务中。

## Dual Encoder文本表示模型

基于监督数据训练的文本表示模型通常采用Dual Encoder框架, 如下图所示。在Dual Encoder框架中, Query和Document文本通过预训练语言模型编码后, 通常采用预训练语言模型[CLS]位置的向量作为最终的文本向量表示。基于标注数据的标签, 通过计算query-document之间的cosine距离度量两者之间的相关性。

<div align=center><img width="450" height="300" src="./resources/dual-encoder.png" /></div>

## 使用方式和范围

使用方式:
- 直接推理, 对给定文本计算其对应的文本向量表示，向量维度768

使用范围:
- 本模型可以使用在电商领域的文本向量表示及其下游应用场景, 包括双句文本相似度计算、query&多doc候选的相似度排序

### 如何使用

在ModelScope框架上，提供输入文本(默认最长文本长度为128)，即可以通过简单的Pipeline调用来使用coROM文本向量表示模型。ModelScope封装了统一的接口对外提供单句向量表示、双句文本相似度、多候选相似度计算功能

#### 代码示例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = "damo/nlp_corom_sentence-embedding_chinese-tiny-ecom"
pipeline_se = pipeline(Tasks.sentence_embedding,
                       model=model_id)
inputs = {
    'source_sentence': ["阔腿裤女冬牛仔"],
    'sentences_to_compare': [
        "阔腿牛仔裤女秋冬款潮流百搭宽松。",
        "牛仔阔腿裤女大码胖mm高腰显瘦夏季薄款宽松垂感泫雅拖地裤子。",
        "阔腿裤男大码高腰宽松。",
    ]
}
result = pipeline_se(input=inputs)
print (result)

#{'text_embedding': array([[-0.23219466,  0.41309455,  0.26903808, ..., -0.27691665,
#         0.39870635,  0.26265654],
#       [-0.22365344,  0.42022845,  0.26665562, ..., -0.2648437 ,
#         0.4074452 ,  0.27727956],
#       [-0.25315332,  0.38203263,  0.2404599 , ..., -0.3280004 ,
#         0.4147297 ,  0.29768175],
#       [-0.24323429,  0.41074494,  0.24910843, ..., -0.30696353,
#         0.4028608 ,  0.2736367 ],
#       [-0.25041878,  0.3749908 ,  0.24194765, ..., -0.3197235 ,
#         0.41340467,  0.27778074]], dtype=float32), 'scores': [70.26205444335938, 70.#42506408691406, 70.55734252929688, 70.36206817626953]}

# 当输入仅含有soure_sentence时，会输出source_sentence中每个句子的向量表示以及首个句子与其他句子的相似度。
inputs2 = {
    'source_sentence': [
        "阔腿牛仔裤女秋冬款潮流百搭宽松。",
        "牛仔阔腿裤女大码胖mm高腰显瘦夏季薄款宽松垂感泫雅拖地裤子。",
        "阔腿裤男大码高腰宽松。",
    ]
}
result = pipeline_se(input=inputs2)
print (result)
# {'text_embedding': array([[ 0.70111793, -0.09922647, -0.3839505 , ...,  0.04588755,
#        -0.03885759, -0.34192216],
#       [ 0.50684625, -0.24223177, -0.1910337 , ...,  0.19840555,
#        -0.03236133, -0.07606616],
#       [ 0.6396424 , -0.05542405, -0.15083028, ..., -0.2550505 ,
#         0.10299131, -0.23259829]], dtype=float32), 'scores': [68.76390075683594, 67.31983184814453]}
```

**默认向量维度768, scores中的score计算两个向量之间的L2距离得到**

### 模型局限性以及可能的偏差

本模型基于[MultiCPR](https://github.com/Alibaba-NLP/Multi-CPR)(电商领域)上训练，在其他垂类领域文本上的文本效果会有降低，请用户自行评测后决定如何使用

### 训练流程

- 模型: 双塔文本表示模型, 采用coROM模型作为预训练语言模型底座
- 二阶段训练: 模型训练分为两阶段, 一阶段的负样本数据从官方提供文档集随机采样负样本, 二阶段通过Dense Retrieval挖掘难负样本扩充训练训练数据重新训练

模型采用4张NVIDIA V100机器训练, 超参设置如下:
```
train_epochs=3
max_sequence_length=128
batch_size=64
learning_rate=5e-6
optimizer=AdamW
```

### 训练代码示例
```
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
ds = MsDataset.load('dureader-retrieval-ranking', 'zyznull')
train_ds = ds['train'].to_hf_dataset()
dev_ds = ds['dev'].to_hf_dataset()
model_id = 'damo/nlp_corom_sentence-embedding_chinese-tiny-ecom'
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

我们主要在文本向量召回场景下评估模型效果, [MultiCPR](https://github.com/Alibaba-NLP/Multi-CPR)(电商领域)召回评估结果如下:

| Model       | MRR@10 |
|-------------|--------|
| CoROM-Retrieval-base      |  31.85  |
| CoROM-Ranking-base        |  47.28  |
| CoROM-Retrieval-tiny      |  19.53  |
| CoROM-Ranking-tiny        |  39.31  |

## 引用

```BibTeX
@article{Long2022MultiCPRAM,
  title={Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
  author={Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Rui Guo and Jianfeng Xu and Guanjun Jiang and Luxi Xing and P. Yang},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series = {SIGIR 22},
  year={2022}
}
```

#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/nlp_corom_sentence-embedding_chinese-base-ecom.git
```
