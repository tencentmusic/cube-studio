

# StructBERT零样本分类模型介绍

模型详细介绍及实现原理可参考评测文章：[世界那么大，我想去看看——探索ModelScope之零样本分类](https://developer.aliyun.com/article/995247)

Yin等人[1]提出了一种使用预训练的自然语言推理模型来实现零样本分类的方式。
工作原理：将要分类的文本设置为自然语言推理的前提，然后使用每个标签构建一个假设，接着对每个假设进行推理得到文本所属的标签。
该模型可以在不使用下游数据进行训练的情况下，按照指定的标签对文本进行分类。

## 可以直接应用的方向

在文本标注平台上，可使用该零样本分类模型对待标注数据进行预标注，对候选标签进行动态排序，从而提升标注效率。
候选标签越多，可以提升的标注效率越明显。

## 模型描述

该模型使用StructBERT-base在xnli数据集(将英文数据集重新翻译得到中文数据集)上面进行了自然语言推理任务训练。

tiny版本模型：[StructBERT零样本分类-中文-tiny](https://www.modelscope.cn/models/damo/nlp_structbert_zero-shot-classification_chinese-tiny/summary)

base版本模型[**推荐**]：[StructBERT零样本分类-中文-base](https://www.modelscope.cn/models/damo/nlp_structbert_zero-shot-classification_chinese-base/summary)

![模型结构](model.jpg#pic_center)


### 如何使用

在ModelScope框架上，通过调用pipeline，提供待分类的文本以及所有可能的标签即可实现文本分类。

#### 代码范例

```python
from modelscope.pipelines import pipeline

classifier = pipeline('zero-shot-classification', 'damo/nlp_structbert_zero-shot-classification_chinese-large')

labels = ['家居', '旅游', '科技', '军事', '游戏', '故事']
sentence = '世界那么大，我想去看看'
classifier(sentence, candidate_labels=labels)
# {'labels': ['旅游', '故事', '游戏', '家居', '军事', '科技'],
#  'scores': [0.2843151092529297,
#   0.20308202505111694,
#   0.14530399441719055,
#   0.12690572440624237,
#   0.12382000684738159,
#   0.11657321453094482]}
#   预测结果为 "旅游"

classifier(sentence, candidate_labels=labels, multi_label=True)
# {'labels': ['旅游', '故事', '游戏', '科技', '军事', '家居'],
#  'scores': [0.7894195318222046,
#   0.5234490633010864,
#   0.41255447268486023,
#   0.2873048782348633,
#   0.27711278200149536,
#   0.2695293426513672]}
#   如阈值设为0.5，则预测出的标签为 "旅游" 及 "故事"
```

### 模型局限性以及可能的偏差
受训练数据的影响，在不同任务上的性能表现可能会有所差异。

## 训练数据介绍
XNLI是来自MNLI的一个子集，已被翻译成14种不同的语言。

## 模型训练流程

### 预处理
XNLI提供的中文数据集的翻译质量不佳，因此对英文数据集进行了重新翻译。

### 训练
使用经过翻译得到的392462条训练数据对StructBERT-base模型进行了自然语言推理任务的训练。

[//]: # (### 训练代码示范)

[//]: # ()
[//]: # (```python)

[//]: # (import os.path as osp)

[//]: # (from modelscope.trainers import build_trainer)

[//]: # (from modelscope.msdatasets import MsDataset)

[//]: # (from modelscope.utils.hub import read_config)

[//]: # ()
[//]: # (model_id = "damo/nlp_structbert_zero-shot-classification_chinese-large")

[//]: # (dataset_id = 'xnli_zh')

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
在经过翻译得到的5000条测试数据上的f1为82.04。

### 相关论文以及引用信息
```
@article{yin2019benchmarking,
  title={Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach},
  author={Yin, Wenpeng and Hay, Jamaal and Roth, Dan},
  journal={arXiv preprint arXiv:1909.00161},
  year={2019}
}
```