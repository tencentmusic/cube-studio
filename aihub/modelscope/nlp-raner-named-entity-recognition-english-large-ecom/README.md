
# RANER介绍


## What's New
- 2022年12月：
  - 训练所使用的序列理解统一框架[AdaSeq](https://github.com/modelscope/AdaSeq/blob/master/README_zh.md)发布，提供30+ SOTA的复现代码！
  - RaNER家族模型均可在[链接](https://github.com/modelscope/AdaSeq/blob/master/docs/modelcards.md)进行访问！所使用的NER数据集均整理在[链接](https://github.com/modelscope/AdaSeq/blob/master/docs/datasets.md)。


## 模型描述
本方法采用Transformer-CRF模型，使用XLM-Roberta作为预训练模型底座，结合使用外部工具召回的相关句子作为额外上下文，使用Multi-view Training方式进行训练。
模型结构如下图所示：

![模型结构](description/model_image.jpg)

可参考论文：[Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning](https://aclanthology.org/2021.acl-long.142/)


## 期望模型使用方式以及适用范围
本模型主要用于给输入英语句子产出命名实体识别结果。用户可以自行尝试输入英语句子。具体调用方式请参考代码示例。

### 如何使用
在安装ModelScope完成之后即可使用named-entity-recognition(命名实体识别)的能力, 默认单句长度不超过512。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_english-large-ecom')
result = ner_pipeline('pizza shovel')

print(result)
# {'output': [{'type': 'OBJECT_PRODUCT', 'start': 0, 'end': 5, 'span': 'pizza'}, {'type': 'PRODUCT', 'start': 6, 'end': 12, 'span': 'shovel'}]}
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
    model_name_or_path: damo/nlp_raner_named-entity-recognition_english-large-ecom
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
本模型基于ecom-en数据集上训练，在垂类领域英语文本上的NER效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
- ecom-en: 内部英文电商领域数据集

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
| 形状 | SHAPE |
| 店铺 | SHOP |
| 停用词 | STOP |
| 风格 | STYLE |
| 时间 | TIME |

## 数据评估及结果
模型在ecom-en测试数据评估结果:

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| ecom-en | 82.67 | 83.69 | 83.18 |

各个类型的性能如下: 
| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| AUX&PRODUCT | 98.53 | 98.53 | 98.53 |
| BRAND | 88.14 | 84.65 | 86.36 |
| COLOR | 80.0 | 84.21 | 82.05 |
| CONJ | 92.31 | 66.67 | 77.42 |
| CROWD | 89.64 | 91.35 | 90.49 |
| IP | 72.57 | 68.29 | 70.37 |
| LOCATION | 50.0 | 55.56 | 52.63 |
| MAIN_BRAND | 81.56 | 84.88 | 83.19 |
| MATERIAL | 82.09 | 84.66 | 83.36 |
| MEASUREMENT | 87.46 | 91.1 | 89.24 |
| MEASUREMENT&PRODUCT | 91.3 | 94.38 | 92.82 |
| MODEL | 86.89 | 84.85 | 85.86 |
| OBJECT_PRODUCT | 69.57 | 76.35 | 72.8 |
| OCCASION | 74.28 | 72.93 | 73.6 |
| PATTERN | 39.71 | 43.55 | 41.54 |
| PREP | 95.93 | 97.63 | 96.77 |
| PRODUCT | 86.77 | 88.84 | 87.79 |
| SALE | 95.0 | 82.61 | 88.37 |
| SHAPE | 66.53 | 64.43 | 65.46 |
| SHOP | 77.78 | 82.35 | 80.0 |
| STYLE | 76.01 | 79.0 | 77.48 |
| TIME | 96.71 | 98.66 | 97.67 |

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
