
# StructBERT中文文本相似度模型介绍

StructBERT中文文本相似度模型是在structbert-base-chinese预训练模型的基础上，用atec、bq_corpus、chineseSTS、lcqmc、paws-x-zh五个数据集（52.5w条数据，正负比例0.48:0.52）训练出来的相似度匹配模型。由于license权限问题，目前只上传了BQ_Corpus、chineseSTS、LCQMC这三个数据集。

其他数据集：
- ATEC：https://dc.cloud.alipay.com/index#/topic/intro?id=3 
- paws-x-zh：https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition

## 模型描述

模型基于Structbert-tiny-chinese，按照BERT文本对分类的方式，在atec、bq_corpus、chineseSTS、lcqmc、paws-x-zh五个数据集（52.5w条数据）上进行微调。

![模型结构](model.jpg)

## 期望模型使用方式以及适用范围
你可以使用StructBERT中文文本相似度模型，对通用领域的文本相似度任务进行推理。
输入形如（文本A，文本B）的文本对数据，模型会给出该文本对的是否相似的标签（不相似, 相似）以及相应的概率。

### 如何使用
在安装完成ModelScope-lib之后即可使用

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.sentence_similarity, 'damo/nlp_structbert_sentence-similarity_chinese-tiny')
semantic_cls(input=('商务职业学院和财经职业学院哪个好？', '商务职业学院商务管理在哪个校区？'))

```

#### Finetune/训练代码范例
```python
import os.path as osp
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config


model_id = 'damo/nlp_structbert_sentence-similarity_chinese-tiny'
dataset_id = 'afqmc_small'
WORK_DIR = 'workspace'

cfg = read_config(model_id)
cfg.train.max_epochs = 2
cfg.train.work_dir = './workspace'
cfg.train.hooks = cfg.train.hooks = [{
        'type': 'TextLoggerHook',
        'interval': 100
    }]
cfg_file = osp.join(WORK_DIR, 'train_config.json')
cfg.dump(cfg_file)

train_dataset = MsDataset.load(dataset_id, namespace='modelscope', split='train')
eval_dataset = MsDataset.load(dataset_id, namespace='modelscope', split='validation')

kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    cfg_file=cfg_file)


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

1. ATEC：蚂蚁金服比赛数据集。
2. BQ_Corpus：请参考关联数据集DAMO_NLP/BQ_Corpus
3. ChineseSTS：请参考关联数据集DAMO_NLP/ChineseSTS
4. LCQMC：请参考关联数据集DAMO_NLP/LCQMC
5. paws-x-zh：谷歌发布的包含7种语言释义对的数据集，请参考 https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition

## 数据评估及结果

| 数据集   | Avg    | ATEC  | bq_corpus | ChineseSTS | LCQMC | paws-x-zh |
| -------- | ------ | ----- | --------- | ---------- | ----- | --------- |
| Accuracy | 0.8397 | 0.8466 | 0.7714     | 0.9594      | 0.7589 | 0.5573     |


### 相关论文以及引用信息

```bib
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```
