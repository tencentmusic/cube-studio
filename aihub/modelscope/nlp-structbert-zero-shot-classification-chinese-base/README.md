

# StructBERT零样本分类模型介绍

模型详细介绍及实现原理可参考评测文章：[世界那么大，我想去看看——探索ModelScope之零样本分类](https://developer.aliyun.com/article/995247)

Yin等人[1]提出了一种使用预训练的自然语言推理模型来实现零样本分类的方式。
工作原理：将要分类的文本设置为自然语言推理的前提，然后使用每个标签构建一个假设，接着对每个假设进行推理得到文本所属的标签。
该模型可以在不使用下游数据进行训练的情况下，按照指定的标签对文本进行分类。

## 可以直接应用的方向

在文本标注平台上，可使用该零样本分类模型对待标注数据进行自动预标注，对候选标签进行动态排序，从而提升标注效率。
候选标签越多，可以提升的标注效率越明显。

## 模型描述

该模型使用StructBERT-base在xnli数据集(将英文数据集重新翻译得到中文数据集)上面进行了自然语言推理任务训练。

tiny版本模型：[StructBERT零样本分类-中文-tiny](https://www.modelscope.cn/models/damo/nlp_structbert_zero-shot-classification_chinese-tiny/summary)

large版本模型：[StructBERT零样本分类-中文-large](https://www.modelscope.cn/models/damo/nlp_structbert_zero-shot-classification_chinese-large/summary)

![模型结构](model.jpg#pic_center)


### 如何使用

在ModelScope框架上，通过调用pipeline，提供待分类的文本以及所有可能的标签即可实现文本分类。

#### 代码范例

```python
from modelscope.pipelines import pipeline

classifier = pipeline('zero-shot-classification', 'damo/nlp_structbert_zero-shot-classification_chinese-base')

labels = ['家居', '旅游', '科技', '军事', '游戏', '故事']
sentence = '世界那么大，我想去看看'
classifier(sentence, candidate_labels=labels)
# {'labels': ['旅游', '故事', '游戏', '家居', '科技', '军事'],
#  'scores': [0.5115893483161926, 0.16600871086120605, 0.11971449106931686, 0.08431519567966461, 0.06298767030239105, 0.05538451299071312]}

classifier(sentence, candidate_labels=labels, multi_label=True)
# {'labels': ['旅游', '故事', '游戏', '军事', '科技', '家居'],
#  'scores': [0.8916056156158447, 0.4281940162181854, 0.16754530370235443, 0.09658896923065186, 0.08678494393825531, 0.07153557986021042]}
#   如阈值设为0.4，则预测出的标签为 "旅游" 及 "故事"

labels = ['积极', '消极', '中性']
sentence = '世界那么大，我想去看看'
classifier(sentence, candidate_labels=labels)
# {'labels': ['积极', '中性', '消极'],
#  'scores': [0.4817797541618347, 0.38822728395462036, 0.12999308109283447]}
#   预测结果为 "积极"
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
[//]: # (model_id = "damo/nlp_structbert_zero-shot-classification_chinese-base")

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