
# BERT文本相似度-英文-base-学术数据集paws

该模型基于bert-base-uncased，在paws(Paraphrase Adversaries from Word Scrambling)数据集(约9万条)上微调得到。

## 模型描述
![模型结构](model.jpeg)

模型采用英文句子对方式对句子相似度进行学习

## 期望模型使用方式以及适用范围
你可以使用BERT英文文本相似度模型，对通用领域的英文文本相似度任务进行推理。
输入形如（文本A，文本B）的文本对数据，模型会给出该文本对的是否相似的标签（0, 1）以及相应的概率。

### 如何使用
在安装完成ModelScope-lib之后即可使用，请参考[modelscope环境安装](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85) 。

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.sentence_similarity, 'damo/nlp_bert_sentence-similarity_english-base')
semantic_cls(input=('That is a happy person', 'That person is happy'))

```

[//]: # (#### Finetune/训练代码范例)

[//]: # (```python)

[//]: # (import os.path as osp)

[//]: # (from modelscope.trainers import build_trainer)

[//]: # (from modelscope.msdatasets import MsDataset)

[//]: # (from modelscope.utils.hub import read_config)

[//]: # ()
[//]: # ()
[//]: # (model_id = 'damo/nlp_bert_sentence-similarity_english-base')

[//]: # (# use paws dataset)

[//]: # (dataset_id = 'paws')

[//]: # ()
[//]: # (WORK_DIR = 'workspace')

[//]: # ()
[//]: # (cfg = read_config&#40;model_id&#41;)

[//]: # (cfg.train.max_epochs = 2)

[//]: # (cfg.train.work_dir = WORK_DIR)

[//]: # (cfg.train.hooks = cfg.train.hooks = [{)

[//]: # (        'type': 'TextLoggerHook',)

[//]: # (        'interval': 100)

[//]: # (    }])

[//]: # (cfg_file = osp.join&#40;WORK_DIR, 'train_config.json'&#41;)

[//]: # (cfg.dump&#40;cfg_file&#41;)

[//]: # ()
[//]: # (train_dataset = MsDataset.load&#40;dataset_id, namespace='DAMO_NLP', split='train'&#41;)

[//]: # (eval_dataset = MsDataset.load&#40;dataset_id, namespace='DAMO_NLP', split='validation'&#41;)

[//]: # ()
[//]: # (kwargs = dict&#40;)

[//]: # (    model=model_id,)

[//]: # (    train_dataset=train_dataset,)

[//]: # (    eval_dataset=eval_dataset,)

[//]: # (    cfg_file=cfg_file&#41;)

[//]: # ()
[//]: # ()
[//]: # (trainer = build_trainer&#40;default_args=kwargs&#41;)

[//]: # ()
[//]: # (print&#40;'==============================================================='&#41;)

[//]: # (print&#40;'pre-trained model loaded, training started:'&#41;)

[//]: # (print&#40;'==============================================================='&#41;)

[//]: # ()
[//]: # (trainer.train&#40;&#41;)

[//]: # ()
[//]: # (print&#40;'==============================================================='&#41;)

[//]: # (print&#40;'train success.'&#41;)

[//]: # (print&#40;'==============================================================='&#41;)

[//]: # ()
[//]: # (for i in range&#40;cfg.train.max_epochs&#41;:)

[//]: # (    eval_results = trainer.evaluate&#40;f'{WORK_DIR}/epoch_{i+1}.pth'&#41;)

[//]: # (    print&#40;f'epoch {i} evaluation result:'&#41;)

[//]: # (    print&#40;eval_results&#41;)

[//]: # ()
[//]: # ()
[//]: # (print&#40;'==============================================================='&#41;)

[//]: # (print&#40;'evaluate success'&#41;)

[//]: # (print&#40;'==============================================================='&#41;)

[//]: # (```)

### 模型局限性以及可能的偏差
模型训练数据有限，在特定行业数据上，效果可能存在一定偏差。


## 数据评估及结果
在测试集上的f1为0.915。

### 相关论文以及引用信息
```
@InProceedings{paws2019naacl,
  title = {{PAWS: Paraphrase Adversaries from Word Scrambling}},
  author = {Zhang, Yuan and Baldridge, Jason and He, Luheng},
  booktitle = {Proc. of NAACL},
  year = {2019}
}
```