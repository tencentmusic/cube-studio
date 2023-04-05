
# RANER介绍

## 模型描述
该模型是基于检索增强(RaNer)方法在西班牙语数据集MultiCoNER-ES-Spanish训练的模型。
本方法采用Transformer-CRF模型，使用XLM-RoBERTa作为预训练模型底座，结合使用外部工具召回的相关句子作为额外上下文，使用Multi-view Training方式进行训练。
模型结构如下图所示：

![模型结构](description/model_image.jpg)

可参考论文：[Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning](https://aclanthology.org/2021.acl-long.142/)


## 期望模型使用方式以及适用范围
本模型主要用于给输入西班牙语句子产出命名实体识别结果。用户可以自行尝试输入西班牙语句子。具体调用方式请参考代码示例。

### 如何使用
在安装ModelScope完成之后即可使用named-entity-recognition(命名实体识别)的能力, 默认单句长度不超过512。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_spanish-large-generic')
result = ner_pipeline('el primer avistamiento por europeos de esta zona fue en 1606 , en la expedición española mandada por luis váez de torres .')

print(result)
# {'output': [{'type': 'LOC', 'start': 80, 'end': 88, 'span': 'española'}, {'type': 'PER', 'start': 101, 'end': 120, 'span': 'luis váez de torres'}]}
```


#### 基于AdaSeq进行微调和推理（仅需一行命令）
**AdaSeq**是一个基于ModelScope的一站式NLP序列理解开源工具箱，支持高效训练自定义模型，旨在提高开发者和研究者们的开发和创新效率，助力模型快速定制和前沿论文工作落地。

1. 安装AdaSeq

```shell
pip install adaseq
```

2. 模型微调

准备训练配置，将下面的代码保存为train.yaml。

该配置中的数据集为示例数据集toy_msra，如需使用自定义数据或调整参数，可参考《[AdaSeq模型训练最佳实践](https://github.com/modelscope/AdaSeq/blob/master/docs/tutorials/training_a_model_zh.md)》，准备数据或修改配置文件。AdaSeq中也提供了大量的[模型、论文、比赛复现示例]([https://github.com/modelscope/AdaSeq/tree/master/examples](https://github.com/modelscope/AdaSeq/tree/master/examples))，欢迎大家使用。

```yaml
experiment:
  exp_dir: experiments/
  exp_name: toy_msra
  seed: 42

task: named-entity-recognition

dataset:
  name: damo/toy_msra

preprocessor:
  type: sequence-labeling-preprocessor
  max_length: 100

data_collator: SequenceLabelingDataCollatorWithPadding

model:
  type: sequence-labeling-model
  embedder:
    model_name_or_path: damo/nlp_raner_named-entity-recognition_spanish-large-generic
  dropout: 0.1
  use_crf: true

train:
  max_epochs: 5
  dataloader:
    batch_size_per_gpu: 8
  optimizer:
    type: AdamW
    lr: 5.0e-5
    param_groups:
      - regex: crf
        lr: 5.0e-1
    options:
      cumulative_iters: 4

evaluation:
  dataloader:
    batch_size_per_gpu: 16
  metrics:
    - type: ner-metric
```

运行命令开始训练。在GPU上训练需要至少6G显存，可以根据实际GPU情况调整batch_size等参数。

```shell
adaseq train -c train.yaml
```

3. 模型推理

模型会保存在 `./experiments/toy_msra/${yymmddHHMMSS.ffffff}/output/`

可以将上文推理示例代码中的model_id替换为本地路径（绝对路径）进行推理

保存的模型也可上传到ModelScope进行使用

### 模型局限性以及可能的偏差
本模型基于MultiCoNER-ES-Spanish数据集上训练，在垂类领域西班牙语文本上的NER效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
- [MultiCoNER-ES-Spanish](https://www.amazon.science/publications/semeval-2022-task-11-multilingual-complex-named-entity-recognition-multiconer) 通用领域西班牙语命名实体识别公开数据集，包括地名, 人名, 公司名, 创作名, 消费品, 其他组织名，共235587个句子。


| 实体类型 | 英文名 |
|--------|--------|
| 公司名 | CORP |
| 创作名 | CW |
| 其他组织名 | GRP |
| 地名 | LOC |
| 人名 | PER |
| 消费品 | PROD |

## 数据评估及结果
模型在MultiCoNER-ES-Spanish验证数据评估结果:

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| MultiCoNER-ES-Spanish | 94.72 | 94.56 | 94.64 |

各个类型的性能如下: 
| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| CORP | 96.35 | 93.62 | 94.96 |
| CW | 92.11 | 91.15 | 91.62 |
| GRP | 91.91 | 94.64 | 93.26 |
| LOC | 96.35 | 96.35 | 96.35 |
| PER | 95.18 | 95.95 | 95.56 |
| PROD | 96.03 | 94.16 | 95.08 |

### 相关论文以及引用信息
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@inproceedings{wang-etal-2021-improving,
    title = "Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning",
    author = "Wang, Xinyu  and
      Jiang, Yong  and
      Bach, Nguyen  and
      Wang, Tao  and
      Huang, Zhongqiang  and
      Huang, Fei  and
      Tu, Kewei",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.142",
    pages = "1800--1812",
}

@inproceedings{wang-etal-2022-damo,
    title = "{DAMO}-{NLP} at {S}em{E}val-2022 Task 11: A Knowledge-based System for Multilingual Named Entity Recognition",
    author = "Wang, Xinyu  and
      Shen, Yongliang  and
      Cai, Jiong  and
      Wang, Tao  and
      Wang, Xiaobin  and
      Xie, Pengjun  and
      Huang, Fei  and
      Lu, Weiming  and
      Zhuang, Yueting  and
      Tu, Kewei  and
      Lu, Wei  and
      Jiang, Yong",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.200",
    pages = "1457--1468",
}
```
