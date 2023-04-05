
# RANER介绍

## 模型描述
本方法采用Transformer-CRF模型，使用XLM-Roberta作为预训练模型底座，结合使用外部工具召回的相关句子作为额外上下文，使用Multi-view Training方式进行训练。
模型结构如下图所示：

![模型结构](description/model_image.jpg)

可参考论文：[Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning](https://aclanthology.org/2021.acl-long.142/)


## 期望模型使用方式以及适用范围
本模型主要用于给输入法语句子产出命名实体识别结果。用户可以自行尝试输入法语句子。具体调用方式请参考代码示例。

### 如何使用
在安装ModelScope完成之后即可使用named-entity-recognition(命名实体识别)的能力, 默认单句长度不超过512。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_french-large-ecom')
result = ner_pipeline('voile de mariage')

print(result)
# {'output': [{'type': 'PRODUCT', 'start': 0, 'end': 5, 'span': 'voile'}, {'type': 'PREP', 'start': 6, 'end': 8, 'span': 'de'}, {'type': 'OCCASION', 'start': 9, 'end': 16, 'span': 'mariage'}]}
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
    model_name_or_path: damo/nlp_raner_named-entity-recognition_french-large-ecom
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
本模型基于ecom-fr数据集上训练，在垂类领域法语文本上的NER效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
- ecom-fr: 内部英文电商领域数据集

| 实体类型 | 英文名 |
|----------|--------|
| 宽泛的名词 | AUX |
| 宽泛的名词&产品词 | AUX&PRODUCT |
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
| 产品词&宽泛的名词 | PRODUCT&AUX |
| 产品词&度量值 | PRODUCT&MEASUREMENT |
| 销售词 | SALE |
| 形状 | SHAPE |
| 停用词 | STOP |
| 风格 | STYLE |
| 时间 | TIME |

## 数据评估及结果
模型在ecom-fr测试数据评估结果:

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| ecom-fr | 87.02 | 87.35 | 87.18 |

各个类型的性能如下: 
| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| AUX | 33.33 | 25.0 | 28.57 |
| AUX&PRODUCT | 44.44 | 26.67 | 33.33 |
| BRAND | 89.33 | 83.75 | 86.45 |
| COLOR | 79.75 | 88.73 | 84.0 |
| CONJ | 81.25 | 100.0 | 89.66 |
| CROWD | 93.67 | 93.27 | 93.47 |
| IP | 79.63 | 82.17 | 80.88 |
| LOCATION | 94.12 | 84.21 | 88.89 |
| MAIN_BRAND | 78.21 | 80.62 | 79.39 |
| MATERIAL | 84.29 | 88.57 | 86.38 |
| MEASUREMENT | 88.69 | 88.17 | 88.43 |
| MEASUREMENT&PRODUCT | 75.0 | 37.5 | 50.0 |
| MODEL | 80.51 | 80.51 | 80.51 |
| OBJECT_PRODUCT | 75.23 | 75.94 | 75.59 |
| OCCASION | 80.68 | 80.62 | 80.65 |
| PATTERN | 65.0 | 56.52 | 60.47 |
| PREP | 93.56 | 94.21 | 93.89 |
| PRODUCT | 92.9 | 93.19 | 93.04 |
| PRODUCT&AUX | 86.67 | 93.69 | 90.04 |
| PRODUCT&MEASUREMENT | 82.93 | 89.47 | 86.08 |
| SALE | 70.0 | 100.0 | 82.35 |
| SHAPE | 80.0 | 78.26 | 79.12 |
| STOP | 100.0 | 80.0 | 88.89 |
| STYLE | 82.15 | 79.43 | 80.77 |
| TIME | 95.6 | 97.75 | 96.67 |

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
