
# StructBERT中文情感分类模型介绍

情感分类任务，通常为输入一段句子或一段话，返回该段话正向/负向的情感极性，在用户评价，观点抽取，意图识别中往往起到重要作用。StructBERT中文情感分类模型是基于bdci、dianping、jd binary、waimai-10k四个数据集（11.5w条数据）训练出来的情感分类模型。由于license权限问题，目前只上传了dianping和jd binary两个数据集。

其他数据集：
- bdci：https://www.datafountain.cn/competitions/310/datasets
- waimai-10k：https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/waimai_10k



## 模型描述

模型基于Structbert-large-chinese，在bdci、dianping、jd binary、waimai-10k四个数据集（11.5w条数据）上fine-tune得到。

![模型结构](model.jpg)

## 期望模型使用方式以及适用范围

你可以使用StructBERT中文情感分类模型模型，对通用领域的中文情感分类任务进行推理。
输入自然语言文本，模型会给出该文本的情感分类标签(0, 1)以及相应的概率。

### 如何使用
在安装完成ModelScope-lib之后即可使用

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_structbert_sentiment-classification_chinese-large')
semantic_cls(input='启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音')
```

#### 微调代码范例
```python
import os.path as osp
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config
from modelscope.metainfo import Metrics


model_id = 'damo/nlp_structbert_sentiment-classification_chinese-large'
dataset_id = 'jd'

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
            'labels': ['负面', '正面'],
            'first_sequence': 'sentence',
            'label': 'label',
        }
    }
    return cfg


train_dataset = MsDataset.load(dataset_id, namespace='DAMO_NLP', split='train').to_hf_dataset()
eval_dataset = MsDataset.load(dataset_id, namespace='DAMO_NLP', split='validation').to_hf_dataset()

# remove useless case
train_dataset = train_dataset.filter(lambda x: x["label"] != None and x["sentence"] != None)
eval_dataset = eval_dataset.filter(lambda x: x["label"] != None and x["sentence"] != None)

# map float to index
def map_labels(examples):
    map_dict = {0: "负面", 1: "正面"}
    examples['label'] = map_dict[int(examples['label'])]
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

### 模型局限性以及可能的偏差
模型训练数据有限，效果可能存在一定偏差。

## 训练数据介绍

1. bdci：CCF汽车行业用户观点主题及情感识别比赛数据集。
2. dianping：请参考关联数据集DAMO_NLP/yf_dianping
3. jd binary：请参考关联数据集DAMO_NLP/jd
4. waimai-10k：某外卖平台收集的用户评价，正向4000条，负向约8000条

数据来源于https://github.com/CLUEbenchmark/CLUEDatasetSearch

## 数据评估及结果

| 数据集   | BDCI2018 | Dianping | JD Binary | Waimai-10k |
| -------- | -------- | -------- | --------- | ---------- |
| Accuracy | 0.8596    | 0.7725    | 0.92     | 0.9154      |


```bib
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```
