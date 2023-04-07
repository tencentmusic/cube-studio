

# StructBERT中文自然语言推理模型介绍

自然语言推理任务（NLI）通常指判断一对句子对（前提句，假设句）在语义上是否存在推理蕴涵关系。作为自然语言理解的一个重要组成部分,NLI专注于语义理解，是一项分类任务。
StructBERT中文自然语言推理模型是在structbert-base-chinese预训练模型的基础上，用CMNLI、OCNLI两个数据集（45.8w条数据）训练出来的自然语言推理模型。

## 模型描述

模型基于Structbert-tiny-chinese，按照BERT论文中的方式，在CMNLI、OCNLI两个数据集（45.8w条数据）上fine-tune得到。

![模型结构](model.jpg)

## 期望模型使用方式以及适用范围

你可以使用StructBERT中文自然语言推理模型，对通用领域的自然语言推理任务进行推理。
输入形如（前提句，假设句）的句子对数据，模型会给出该句子对应的自然语言推理标签 {"矛盾": 0, "蕴涵": 1, "中立": 2} 以及相应的概率。

### 如何使用

在安装完成ModelScope-lib，请参考  [modelscope环境安装](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85) 。

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.nli, 'damo/nlp_structbert_nli_chinese-tiny')
semantic_cls(input=('一月份跟二月份肯定有一个月份有.', '肯定有一个月份有'))

```

#### 微调代码范例
```
pip install datasets==2.1.0
```

```python
import os.path as osp
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config
from modelscope.metainfo import Metrics
from modelscope.utils.constant import DownloadMode

model_id = 'damo/nlp_structbert_nli_chinese-tiny'
dataset_id = 'clue'

WORK_DIR = 'workspace'

max_epochs = 2
def cfg_modify_fn(cfg):
    cfg.train.max_epochs = max_epochs
    cfg.train.hooks = cfg.train.hooks = [{
            'type': 'TextLoggerHook',
            'interval': 100
        }]
    cfg.evaluation.metrics = [Metrics.seq_cls_metric]
    cfg['dataset'] = {
        'train': {
            'first_sequence': 'sentence1',
            'second_sequence': 'sentence2',
            'label': 'label',
        }
    }
    return cfg


train_dataset = MsDataset.load(dataset_id, namespace='modelscope', subset_name='ocnli', split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD).to_hf_dataset()
eval_dataset = MsDataset.load(dataset_id, namespace='modelscope', subset_name='ocnli', split='validation', download_mode=DownloadMode.FORCE_REDOWNLOAD).to_hf_dataset()

# # remove useless case
# train_dataset = train_dataset.filter(lambda x: x["label"] != None and x["sentence"] != None)
# eval_dataset = eval_dataset.filter(lambda x: x["label"] != None and x["sentence"] != None)

# map float to index
def map_labels(examples):
    map_dict = {0: "矛盾", 1: "蕴涵", 2: "中立"}
    examples['label'] = map_dict.get(int(examples['label']), "中立")
    return examples

train_dataset = train_dataset.map(map_labels)
eval_dataset = eval_dataset.map(map_labels)

kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=WORK_DIR,
    cfg_modify_fn=cfg_modify_fn)


trainer = build_trainer(name='nlp-base-trainer', default_args=kwargs)

print('===============================================================')
print('pre-trained model loaded, training started:')
print('===============================================================')

trainer.train()

print('===============================================================')
print('train success.')
print('===============================================================')

for i in range(max_epochs):
    eval_results = trainer.evaluate(f'{WORK_DIR}/epoch_{i+1}.pth')
    print(f'epoch {i} evaluation result:')
    print(eval_results)


print('===============================================================')
print('evaluate success')
print('===============================================================')
```


#### 模型局限性以及可能的偏差
模型训练数据有限，不能包含所有行业，因此在特定行业数据上，效果可能存在一定偏差。

## 训练数据介绍

1. CMNLI（Chinese Multi-Genre NLI）：用于判断给定的两个句子之间属于蕴涵、中立、矛盾关系。数据来源于fiction， telephone，travel，government等。数据集由经过中英转化的XNLI与MNLI组成，
   dev由XNLI中的dev和MNLI中的matched组成，test由XNLI中的test和MNLI中的mismatched组成；train/dev/test的数据量分别是391782/12426/13880。

2. OCNLI（Original Chinese Natural Language Inference）：原生中文自然语言推理数据集，是第一个非翻译的、使用原生汉语的大型中文自然语言推理数据集；train/dev/test对应的数据量分别是
   50k/3k/3k。

数据来源于https://github.com/CLUEbenchmark/CLUEDatasetSearch


## 数据评估及结果

| 数据集   | CMNLI | OCNLI |
| -------- | ----- | ----- |
| Accuracy | 0.6556 | 0.6173 |

### 相关论文以及引用信息

```bib
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```