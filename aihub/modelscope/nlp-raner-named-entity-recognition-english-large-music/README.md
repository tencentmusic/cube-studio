
# RANER介绍

## 模型描述
本方法采用Transformer-CRF模型，使用xlm-roberta-large作为预训练模型底座，结合使用外部工具召回的相关句子作为额外上下文，使用Multi-view Training方式进行训练。
模型结构如下图所示：

![模型结构](description/model_image.jpg)

可参考论文：[Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning](https://aclanthology.org/2021.acl-long.142/)


## 期望模型使用方式以及适用范围
本模型主要用于给输入英文句子产出命名实体识别结果。用户可以自行尝试输入英文句子。具体调用方式请参考代码示例。

### 如何使用
在安装ModelScope完成之后即可使用named-entity-recognition(命名实体识别)的能力, 默认单句长度不超过512。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_english-large-music')
result = ner_pipeline('Brad Wilk of Rage Against the Machine was chosen as the drummer , and Rick Rubin was chosen as the producer .')

print(result)
# {'output': [{'type': 'musicalartist', 'start': 0, 'end': 9, 'span': 'Brad Wilk'}, {'type': 'band', 'start': 13, 'end': 37, 'span': 'Rage Against the Machine'}, {'type': 'musicalartist', 'start': 70, 'end': 80, 'span': 'Rick Rubin'}]}
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
    model_name_or_path: damo/nlp_raner_named-entity-recognition_english-large-music
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
本模型基于music数据集上训练，在垂类领域英文文本上的NER效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
- [Music](https://github.com/zliucr/CrossNER) 音乐领域英文命名实体识别公开数据集，包括13种实体类型，共100个句子。

| 实体类型 | 英文名 |
|----------|--------|
| 专辑名 | album |
| 奖项名 | award |
| 乐队名 | band |
| 国家名 | country |
| 事件名 | event |
| 位置名 | location |
| 杂项 | misc |
| 音乐家 | musicalartist |
| 乐器名 | musicalinstrument |
| 音乐流派 | musicgenre |
| 组织名 | organisation |
| 人名 | person |
| 歌名 | song |

## 数据评估及结果
模型在music测试数据评估结果:

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| music | 80.48 | 80.34 | 80.41 |

各个类型的性能如下: 
| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| album | 73.98 | 87.35 | 80.11 |
| award | 83.72 | 83.08 | 83.4 |
| band | 84.0 | 86.36 | 85.17 |
| country | 80.21 | 93.75 | 86.46 |
| event | 66.67 | 66.67 | 66.67 |
| location | 89.03 | 82.8 | 85.8 |
| misc | 41.11 | 21.64 | 28.35 |
| musicalartist | 83.54 | 91.65 | 87.41 |
| musicalinstrument | 33.33 | 4.55 | 8.0 |
| musicgenre | 79.01 | 88.09 | 83.3 |
| organisation | 80.88 | 79.33 | 80.1 |
| person | 84.38 | 33.75 | 48.21 |
| song | 84.97 | 72.25 | 78.1 |

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
