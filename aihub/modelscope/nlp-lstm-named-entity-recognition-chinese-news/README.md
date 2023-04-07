
# 模型介绍

## 模型描述

本方法采用char-BiLSTM-CRF模型，word-embedding使用[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)

模型结构可参考论文：[Neural Architectures for Named Entity Recognition
](https://aclanthology.org/N16-1030.pdf)

## 期望模型使用方式以及适用范围
本模型主要用于给输入中文句子产出命名实体识别结果。用户可以自行尝试输入中文句子。具体调用方式请参考代码示例。

### 如何使用
在安装ModelScope完成之后即可使用named-entity-recognition(命名实体识别)的能力, 默认单句长度不超过512。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_lstm_named-entity-recognition_chinese-news')
result = ner_pipeline('（新华社北京２月２８日电）')

print(result)
# {'output': [{'type': 'ORG', 'start': 0, 'end': 3, 'span': '新华社'},{'type': 'LOC', 'start': 3, 'end': 5, 'span': '北京'}]}
```


### 模型局限性以及可能的偏差
本模型基于msra数据集(新闻领域)上训练，在垂类领域中文文本上的NER效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
- [MSRA](https://aclanthology.org/W06-0115/) 新闻领域中文命名实体识别公开数据集，包括地缘政治实体、地名、机构名、人名，共50729个句子。

| 实体类型 | 英文名 |
|----------|--------|
| 地名     | LOC    |
| 机构名   | ORG   |
| 人名   |  PER  |


## 数据评估及结果
模型在MSRA测试数据评估结果:

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| MSRA | 91.00 | 89.26 | 90.12 |

各个类型的性能如下: 

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| LOC | 94.23 | 90.33 | 92.24 |
| ORG | 84.30 | 86.39 | 85.33 |
| PER | 91.15 | 89.70 | 90.42 |
