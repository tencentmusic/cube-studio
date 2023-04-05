
# RANER介绍
## What's New


- 2023年2月：
 - - 如当前模型不满足您的需求，请尝试信息抽取快速定制能力。具体可以体验我们的[创空间](https://www.modelscope.cn/studios/TTCoding/maoe_fsl_ner/summary)，零代码仅需要一~五条标注样本即可定制一个NER模型！

- 2022年12月：
  - 训练所使用的序列理解统一框架[AdaSeq](https://github.com/modelscope/AdaSeq/blob/master/README_zh.md)发布，提供30+ SOTA的复现代码！
  - RaNER家族模型均可在[链接](https://github.com/modelscope/AdaSeq/blob/master/docs/modelcards.md)进行访问！所使用的NER数据集均整理在[链接](https://github.com/modelscope/AdaSeq/blob/master/docs/datasets.md)。

## 模型描述
本方法采用Transformer-CRF模型，使用StructBERT作为预训练模型底座，结合使用外部工具召回的相关句子作为额外上下文，使用Multi-view Training方式进行训练。
模型结构如下图所示：

![模型结构](description/model_image.jpg)

可参考论文：[Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning](https://aclanthology.org/2021.acl-long.142/)


## 期望模型使用方式以及适用范围
本模型主要用于给输入中文句子产出命名实体识别结果。用户可以自行尝试输入中文句子。具体调用方式请参考代码示例。

### 如何使用
在安装ModelScope完成之后即可使用named-entity-recognition(命名实体识别)的能力, 默认单句长度不超过512。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_raner_named-entity-recognition_chinese-base-ecom-50cls')
result = ner_pipeline('eh 摇滚狗涂鸦拔印宽松牛仔裤 情侣款')

print(result)
# {'output': [{'type': '品牌', 'start': 0, 'end': 2, 'span': 'eh'}, {'type': '品牌', 'start': 3, 'end': 6, 'span': '摇滚狗'}, {'type': '款式_其他', 'start': 6, 'end': 8, 'span': '涂鸦'}, {'type': '款式_其他', 'start': 8, 'end': 10, 'span': '拔印'}, {'type': '款式_其他', 'start': 10, 'end': 12, 'span': '宽松'}, {'type': '材质_面料', 'start': 12, 'end': 14, 'span': '牛仔'}, {'type': '产品_核心产品', 'start': 14, 'end': 15, 'span': '裤'}, {'type': '款式_其他', 'start': 16, 'end': 19, 'span': '情侣款'}]}
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
    model_name_or_path: damo/nlp_raner_named-entity-recognition_chinese-base-ecom-50cls
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
本模型基于ecom-cn-50cls数据集上训练，在垂类领域中文文本上的NER效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
- ecom-cn-50cls: 内部中文细粒度电商领域数据集，共包含54类实体，具体类型参见下表。

## 数据评估及结果
模型在ecom-cn-50cls测试数据评估结果:

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| ecom-cn-50cls | 73.19 | 75.21 | 74.18 |

各个类型的性能如下: 
| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 产品_修饰产品 | 26.23 | 10.88 | 15.38 |
| 产品_其他 | 71.39 | 78.07 | 74.58 |
| 产品_核心产品 | 77.45 | 80.25 | 78.82 |
| 人名_真实人名 | 63.16 | 85.71 | 72.73 |
| 人名_虚拟角色 | 0.0 | 0.0 | 0.0 |
| 使用方式_安装方式 | 0.0 | 0.0 | 0.0 |
| 使用方式_穿着方式 | 0.0 | 0.0 | 0.0 |
| 使用方法_其他 | 0.0 | 0.0 | 0.0 |
| 修饰_产品属性 | 61.54 | 61.32 | 61.43 |
| 修饰_其他 | 51.14 | 47.46 | 49.23 |
| 修饰_口味 | 57.35 | 63.93 | 60.47 |
| 修饰_外观描述 | 52.98 | 56.06 | 54.48 |
| 修饰_工作方式 | 58.82 | 55.56 | 57.14 |
| 修饰_评价体验 | 0.0 | 0.0 | 0.0 |
| 功能功效 | 72.39 | 79.86 | 75.94 |
| 品牌 | 78.19 | 78.92 | 78.55 |
| 地点地域_产地 | 41.38 | 81.82 | 54.96 |
| 地点地域_其他 | 45.45 | 23.81 | 31.25 |
| 地点地域_发货地 | 0.0 | 0.0 | 0.0 |
| 地点地域_商标产地 | 0.0 | 0.0 | 0.0 |
| 地点地域_适用地区 | 0.0 | 0.0 | 0.0 |
| 型号 | 44.05 | 50.89 | 47.23 |
| 尺寸规格_其他 | 80.36 | 90.0 | 84.91 |
| 尺寸规格_售卖规格 | 73.31 | 73.31 | 73.31 |
| 尺寸规格_外观尺寸 | 76.37 | 76.63 | 76.5 |
| 尺寸规格_指标参数 | 39.33 | 42.17 | 40.7 |
| 尺寸规格_重量 | 86.14 | 88.78 | 87.44 |
| 工艺 | 68.64 | 80.2 | 73.97 |
| 文化作品_书名 | 45.95 | 57.95 | 51.26 |
| 文化作品_其他 | 0.0 | 0.0 | 0.0 |
| 文化作品_影视名称 | 0.0 | 0.0 | 0.0 |
| 文化作品_游戏名称 | 0.0 | 0.0 | 0.0 |
| 材质_其他 | 70.59 | 68.57 | 69.57 |
| 材质_木质材质 | 88.66 | 89.58 | 89.12 |
| 材质_金属材质 | 60.61 | 65.93 | 63.16 |
| 材质_面料 | 83.4 | 88.52 | 85.88 |
| 款式_其他 | 77.37 | 82.39 | 79.8 |
| 款式_厚薄 | 90.67 | 87.18 | 88.89 |
| 款式_袖型 | 94.26 | 96.64 | 95.44 |
| 款式_裙型 | 0.0 | 0.0 | 0.0 |
| 款式_裤型 | 0.0 | 0.0 | 0.0 |
| 款式_鞋型 | 0.0 | 0.0 | 0.0 |
| 款式_领型 | 95.71 | 98.53 | 97.1 |
| 系列 | 33.46 | 29.72 | 31.48 |
| 组织机构 | 62.5 | 83.33 | 71.43 |
| 适用范围_其他 | 66.25 | 64.81 | 65.52 |
| 适用范围_适用人群 | 88.68 | 89.05 | 88.87 |
| 适用范围_适用场景 | 73.49 | 79.21 | 76.24 |
| 适用范围_适用季节 | 86.19 | 82.21 | 84.15 |
| 适用范围_适用对象 | 73.09 | 80.25 | 76.51 |
| 颜色_其他 | 73.47 | 81.82 | 77.42 |
| 颜色_色彩 | 85.04 | 84.05 | 84.54 |
| 颜色_配色方案 | 0.0 | 0.0 | 0.0 |
| 风格 | 91.08 | 93.76 | 92.4 |

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
