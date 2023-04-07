

# 零样本分类模型介绍

模型详细介绍及实现原理可参考评测文章：[世界那么大，我想去看看——探索ModelScope之零样本分类](https://developer.aliyun.com/article/995247)

Yin等人[1]提出了一种使用预训练的自然语言推理模型来实现零样本分类的方式。
工作原理：将要分类的文本设置为自然语言推理的前提，然后使用每个标签构建一个假设，接着对每个假设进行推理得到文本所属的标签。
该模型可以在不使用下游数据进行训练的情况下，按照指定的标签对文本进行分类。

![模型结构](model_en.jpg#pic_center)

## 模型描述

该模型使用bert-base-uncased在multi_nli数据集上面进行了自然语言推理任务训练。

### 如何使用

在ModelScope框架上，通过调用pipeline，提供待分类的文本以及所有可能的标签即可实现文本分类。

#### 代码范例

```python
from modelscope.pipelines import pipeline

classifier = pipeline('zero-shot-classification', 'damo/nlp_bert_zero-shot_english-base')

sentence = 'one day I will see the world'
labels = ['travel', 'cooking', 'dancing']

classifier(sentence, candidate_labels=labels)
# {'labels': ['travel', 'cooking', 'dancing'], 
# 'scores': [0.4974762201309204, 0.2736286520957947, 0.22889523208141327]}

labels = ['travel', 'cooking', 'dancing', 'exploration']
classifier(sentence, candidate_labels=labels, multi_label=True)
# {'labels': ['exploration', 'travel', 'dancing', 'cooking'], 
# 'scores': [0.5681723356246948, 0.49103984236717224, 0.08524788916110992, 0.04436328634619713]}
```

### 模型局限性以及可能的偏差
受训练数据的影响，在不同任务上的性能表现可能会有所差异。

[//]: # (## 训练数据介绍)


[//]: # (### 训练代码示范)

[//]: # ()
[//]: # (```python)

[//]: # (import os.path as osp)

[//]: # (from modelscope.trainers import build_trainer)

[//]: # (from modelscope.msdatasets import MsDataset)

[//]: # (from modelscope.utils.hub import read_config)

[//]: # ()
[//]: # (model_id = "damo/nlp_bert_zero-shot_english-base")

[//]: # (dataset_id = 'mnli')

[//]: # (WORK_DIR = 'workspace')

[//]: # ()
[//]: # (# todo: not sure)

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
[//]: # ()
[//]: # (train_dataset = MsDataset.load&#40;dataset_id, namespace='Yixiang', split='train'&#41;)

[//]: # (eval_dataset = MsDataset.load&#40;dataset_id, namespace='Yixiang', split='validation'&#41;)

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

## 数据评估及结果
在multi_nli的验证集上的f1为84.87。

### 相关论文以及引用信息
```
@article{yin2019benchmarking,
  title={Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach},
  author={Yin, Wenpeng and Hay, Jamaal and Roth, Dan},
  journal={arXiv preprint arXiv:1909.00161},
  year={2019}
}
```