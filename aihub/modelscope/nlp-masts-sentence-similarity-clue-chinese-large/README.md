
# MaSTS中文文本相似度-CLUE语义匹配模型介绍

MaSTS中文文本相似度-CLUE语义匹配模型是基于[MaSTS预训练模型-CLUE语义匹配](https://modelscope.cn/models/damo/nlp_masts_backbone_clue_chinese-large/summary)，在[QBQTC数据集](https://modelscope.cn/datasets/damo/QBQTC/summary)上训练得到的相似度匹配模型。通过集成此模型在[CLUE语义匹配榜](https://www.cluebenchmarks.com/sim.html)上获得了第一名的成绩。

使用教程请参考 https://developer.aliyun.com/article/1128425 和Jupyter Notebook`tutorial.ipynb`。

## 模型描述

模型按照BERT文本对分类的方式，在QBQTC数据集上进行微调。

### 期望模型使用方式以及适用范围

输入形如（文本A，文本B）的文本对数据，模型会给出该文本对相关性的标签（“0”，"1"，"2"）以及相应的概率。相关性的含义：0，相关程度差；1，有一定相关性；2，非常相关。数字越大相关性越高。

### 模型局限性以及可能的偏差

模型训练数据有限，在其他数据上效果可能存在一定偏差。

## 如何使用

### 环境安装

请参考ModelScope[环境安装](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)。

### 推理代码范例

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


similarity_pipeline = pipeline(Tasks.sentence_similarity, 'damo/nlp_masts_sentence-similarity_clue_chinese-large', model_revision='v1.0.0')
similarity_pipeline(input=('小孩咳嗽感冒', '小孩感冒过后久咳嗽该吃什么药育儿问答宝宝树'))
```

### Finetune/训练代码范例

```python
import os.path as osp
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config


model_id = 'damo/nlp_masts_backbone_clue_chinese-large'
dataset_id = 'QBQTC'

WORK_DIR = 'workspace'

cfg = read_config(model_id, revision='v1.0.0')
cfg.train.work_dir = WORK_DIR
cfg_file = osp.join(WORK_DIR, 'train_config.json')
cfg.dump(cfg_file)

train_dataset = MsDataset.load(dataset_id, namespace='damo', subset_name='default', split='train', keep_default_na=False)
eval_dataset = MsDataset.load(dataset_id, namespace='damo', subset_name='public', split='test', keep_default_na=False)

kwargs = dict(
    model=model_id,
    model_revision='v1.0.0',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    cfg_file=cfg_file,
)

trainer = build_trainer(default_args=kwargs)

print('===============================================================')
print('pre-trained model loaded, training started:')
print('===============================================================')

trainer.train()

print('===============================================================')
print('train success.')
print('===============================================================')

for i in range(cfg.train.max_epochs):
    eval_results = trainer.evaluate(f'{WORK_DIR}/epoch_{i+1}.pth')
    print(f'epoch {i} evaluation result:')
    print(eval_results)

print('===============================================================')
print('evaluate success')
print('===============================================================')
```

## 数据评估及结果

| Dataset                  | Marco F1 | Accuracy |
| ------------------------ | -------- | -------- |
| 公开测试集（test_public) | 74.1     | 79.7     |

![榜单](./leaderboard.png)
