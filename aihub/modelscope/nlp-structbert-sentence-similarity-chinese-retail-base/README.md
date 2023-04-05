
# 电商领域StructBERT中文文本相似度模型介绍
电商领域StructBERT中文文本相似度模型是在structbert-base-chinese预训练模型的基础上，使用电商领域标注数据训练出来的相似度匹配模型，特别适用于电商领域智能客服FAQ匹配，也可以用作其他相似度计算任务。

## 模型描述

模型基于Structbert-base-chinese，按照BERT文本对分类的方式，在电商领域数据上进行微调得到电商领域StructBERT文本相似度模型。

![模型结构](model.jpg)

## 期望模型使用方式以及适用范围
你可以使用电商领域StructBERT中文文本相似度模型，对电商领域的文本相似度任务进行推理。
输入形如（文本A，文本B）的文本对数据，模型会给出该文本对的是否相似的标签（不相似, 相似）以及相应的概率。

### 如何使用

#### 环境安装
在安装完成ModelScope-lib，请参考  [modelscope环境安装](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85) 。

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

similarity_pipeline = pipeline(Tasks.sentence_similarity, 'damo/nlp_structbert_sentence-similarity_chinese-retail-base',model_revision='v1.0.0')
similarity_pipeline(input=('坏了怎么保修', '请问怎样保修的'))

```

#### Finetune/训练代码范例
```python
import os.path as osp
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config


model_id = 'damo/nlp_structbert_sentence-similarity_chinese-retail-base'
# 数据集仅供格式参考，请使用自己的数据
dataset_id = 'BQ_Corpus'

WORK_DIR = 'workspace'

cfg = read_config(model_id,revision='v1.0.0')
cfg.train.max_epochs = 2
cfg.train.work_dir = WORK_DIR
cfg.train.hooks = cfg.train.hooks = [{
        'type': 'TextLoggerHook',
        'interval': 100
    }]
cfg_file = osp.join(WORK_DIR, 'train_config.json')
cfg.dump(cfg_file)

train_dataset = MsDataset.load(dataset_id, namespace='DAMO_NLP', split='train').to_hf_dataset()
eval_dataset = MsDataset.load(dataset_id, namespace='DAMO_NLP', split='validation').to_hf_dataset()

# map float to index
def map_labels(examples):
    map_dict = {0: "不相似", 1: "相似"}
    examples['label'] = map_dict[int(examples['label'])]
    return examples

train_dataset = train_dataset.map(map_labels)
eval_dataset = eval_dataset.map(map_labels)

kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    cfg_file=cfg_file,
    model_revision='v1.0.0')


trainer = build_trainer(name='nlp-base-trainer',default_args=kwargs)

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

### 模型局限性以及可能的偏差
模型训练数据有限，不能包含所有行业，因此在特定行业数据上，效果可能存在一定偏差。

## 训练数据介绍
- 电商域文本匹配数据，由于license问题，暂不公开；

## 数据评估及结果
- 在电商域评测数据中，平均AUC显著高于通用模型；

### 相关论文以及引用信息

```bib
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```
