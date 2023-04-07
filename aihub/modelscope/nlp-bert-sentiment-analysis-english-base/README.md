
# 情感分类模型
本模型是针对实际场景中常见的社交媒体文本情感分析需求所提供的一个模型。模型使用TweetEval社交媒体情感分析文本数据，在BERT预训练模型上进行微调，贴合下游社交媒体领域的情感分类能力。

## 模型描述
针对实际场景中常见的社交媒体文本情感分析需求，达摩院提供了本情感分类模型。该模型通过在BERT预训练模型上使用社交媒体tweet文本情感分类数据微调得到，可用于社交媒体领域的文本情感分类。模型特点如下：

- 本模型基于bert-base-uncased预训练模型进行微调。
- 微调时采用TweetEval社交媒体情感分类数据集。

![model](./model.PNG)

## 期望模型使用方式以及适用范围
输入自然语言文本，模型会给出该文本的情感分类标签(0, 1, 2)，即（Negative, Neutral, Positive）以及相应的概率。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope-lib之后，即可在ModelScope框架上通过Pipeline调用来使用。

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input = 'Good night.'
semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_bert_sentiment-analysis_english-base')
result = semantic_cls(input)

print('输入文本:\n{}\n'.format(input))
print('分类结果:\n{}'.format(result))
```

[//]: # (#### 微调代码范例)

[//]: # (```python)

[//]: # (import os.path as osp)

[//]: # (from modelscope.trainers import build_trainer)

[//]: # (from modelscope.msdatasets import MsDataset)

[//]: # (from modelscope.utils.hub import read_config)

[//]: # (from modelscope.metainfo import Metrics)

[//]: # ()
[//]: # ()
[//]: # (model_id = 'damo/nlp_bert_sentiment-analysis_english-base')

[//]: # (dataset_id = 'jd')

[//]: # ()
[//]: # (WORK_DIR = 'workspace')

[//]: # ()
[//]: # (def cfg_modify_fn&#40;cfg&#41;:)

[//]: # (    cfg.train.max_epochs = 2)

[//]: # (    cfg.train.hooks = cfg.train.hooks = [{)

[//]: # (            'type': 'TextLoggerHook',)

[//]: # (            'interval': 100)

[//]: # (        }])

[//]: # (    cfg.evaluation.metrics = [Metrics.seq_cls_metric])

[//]: # (    cfg['dataset'] = {)

[//]: # (        'train': {)

[//]: # (            'labels': ['0.0', '1,0', '2.0', 'None'],)

[//]: # (            'first_sequence': 'sentence',)

[//]: # (            'label': 'label',)

[//]: # (        })

[//]: # (    })

[//]: # (    return cfg)

[//]: # ()
[//]: # ()
[//]: # (train_dataset = MsDataset.load&#40;dataset_id, namespace='DAMO_NLP', split='train'&#41;.to_hf_dataset&#40;&#41;)

[//]: # (eval_dataset = MsDataset.load&#40;dataset_id, namespace='DAMO_NLP', split='validation'&#41;.to_hf_dataset&#40;&#41;)

[//]: # ()
[//]: # (# remove useless case)

[//]: # (train_dataset = train_dataset.filter&#40;lambda x: x["label"] != None&#41;)

[//]: # (eval_dataset = eval_dataset.filter&#40;lambda x: x["label"] != None&#41;)

[//]: # ()
[//]: # (# map float to index)

[//]: # (def map_labels&#40;examples&#41;:)

[//]: # (    map_dict = {0: "消极", 1: "中立", 2: "积极"})

[//]: # (    examples['label'] = map_dict[int&#40;examples['label']&#41;])

[//]: # (    return examples)

[//]: # ()
[//]: # (train_dataset = train_dataset.map&#40;map_labels&#41;)

[//]: # (eval_dataset = eval_dataset.map&#40;map_labels&#41;)

[//]: # ()
[//]: # (kwargs = dict&#40;)

[//]: # (    model=model_id,)

[//]: # (    train_dataset=train_dataset,)

[//]: # (    eval_dataset=eval_dataset,)

[//]: # (    work_dir=WORK_DIR,)

[//]: # (    cfg_modify_fn=cfg_modify_fn&#41;)

[//]: # ()
[//]: # ()
[//]: # (trainer = build_trainer&#40;name='nlp-base-trainer', default_args=kwargs&#41;)

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
模型在TweetEval情感分类数据集上训练，在其他非社交媒体垂直领域效果可能会有所下降。

## 训练数据介绍
训练数据采用TweetEval情感分类数据集，数据来源于https://github.com/cardiffnlp/tweeteval

## 数据评估及结果
在TweetEval的测试集上的f1为69.18。

### 相关论文以及引用信息
```bib
@article{barbieri2020tweeteval,
  title={Tweeteval: Unified benchmark and comparative evaluation for tweet classification},
  author={Barbieri, Francesco and Camacho-Collados, Jose and Neves, Leonardo and Espinosa-Anke, Luis},
  journal={arXiv preprint arXiv:2010.12421},
  year={2020}
}
```