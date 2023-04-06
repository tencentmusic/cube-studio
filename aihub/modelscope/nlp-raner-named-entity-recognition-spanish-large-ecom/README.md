
# RANER介绍

## 模型描述
本方法采用Transformer-CRF模型，使用XLM-Roberta作为预训练模型底座，结合使用外部工具召回的相关句子作为额外上下文，使用Multi-view Training方式进行训练。
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

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_spanish-large-ecom')
result = ner_pipeline('top cuero')

print(result)
# {'output': [{'type': 'PRODUCT', 'start': 0, 'end': 3, 'span': 'top'}, {'type': 'MATERIAL', 'start': 4, 'end': 9, 'span': 'cuero'}]}
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
    model_name_or_path: damo/nlp_raner_named-entity-recognition_spanish-large-ecom
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
本模型基于ecom-es数据集上训练，在垂类领域西班牙语文本上的NER效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
- ecom-es: 内部西班牙语电商领域数据集

| 实体类型 | 英文名 |
|----------|--------|
| 宽泛的名词 | AUX |
| 宽泛的名词&产品 | AUX&PRODUCT |
| 品牌 | BRAND |
| 颜色 | COLOR |
| 连词 | CONJ |
| 对象 | CROWD |
| IP | IP |
| 地点 | LOCATION |
| 主品牌 | MAIN_BRAND |
| 材质 | MATERIAL |
| 度量值 | MEASUREMENT |
| 度量值&产品 | MEASUREMENT&PRODUCT |
| 型号 | MODEL |
| 产品修饰词 | OBJECT_PRODUCT |
| 适用场景 | OCCASION |
| 图案 | PATTERN |
| 介词 | PREP |
| 产品词 | PRODUCT |
| 销售词 | SALE |
| 形状 | SHAPE |
| 店铺 | SHOP |
| 停用词 | STOP |
| 风格 | STYLE |
| 时间 | TIME |

## 数据评估及结果
模型在ecom-es测试数据评估结果:

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| ecom-es | 90.75 | 91.13 | 90.94 |

各个类型的性能如下: 
| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| AUX | 50.0 | 40.0 | 44.44 |
| AUX&PRODUCT | 95.65 | 98.51 | 97.06 |
| BRAND | 91.43 | 89.39 | 90.4 |
| COLOR | 87.5 | 85.14 | 86.3 |
| CONJ | 97.67 | 95.45 | 96.55 |
| CROWD | 97.65 | 96.56 | 97.1 |
| IP | 83.26 | 74.8 | 78.8 |
| LOCATION | 90.0 | 75.0 | 81.82 |
| MAIN_BRAND | 80.95 | 83.44 | 82.18 |
| MATERIAL | 84.48 | 91.96 | 88.06 |
| MEASUREMENT | 89.19 | 90.55 | 89.86 |
| MEASUREMENT&PRODUCT | 85.71 | 85.71 | 85.71 |
| MODEL | 87.63 | 85.34 | 86.47 |
| OBJECT_PRODUCT | 79.66 | 84.93 | 82.21 |
| OCCASION | 85.97 | 84.1 | 85.02 |
| PATTERN | 73.17 | 61.22 | 66.67 |
| PREP | 97.35 | 98.22 | 97.78 |
| PRODUCT | 95.48 | 96.52 | 96.0 |
| SALE | 80.95 | 89.47 | 85.0 |
| SHAPE | 76.96 | 76.26 | 76.61 |
| STOP | 94.87 | 97.37 | 96.1 |
| STYLE | 75.18 | 77.44 | 76.3 |
| TIME | 96.9 | 99.1 | 97.99 |

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

@inproceedings{zhang-etal-2022-domain,
title = "Domain-Specific NER via Retrieving Correlated Samples",
author = "Zhang, Xin  and
    Yong, Jiang  and
    Wang, Xiaobin  and
    Hu, Xuming  and
    Sun, Yueheng  and
    Xie, Pengjun  and
    Zhang, Meishan",
booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
month = oct,
year = "2022",
address = "Gyeongju, Republic of Korea",
publisher = "International Committee on Computational Linguistics"
}
```
