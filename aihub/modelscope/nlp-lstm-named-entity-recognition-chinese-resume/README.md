
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

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_lstm_named-entity-recognition_chinese-resume')
result = ner_pipeline('常建良，男，')

print(result)
# {'output': [{'type': 'NAME', 'start': 0, 'end': 3, 'span': '常建良'}]}
```


### 模型局限性以及可能的偏差
本模型基于Resume数据集(简历领域)上训练，在垂类领域中文文本上的NER效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
- [Resume](https://aclanthology.org/P18-1144/) 简历领域中文命名实体识别公开数据集，包括国籍、教育背景、籍贯、人名、组织名、专业、民族、职称，共4761个句子。

| 实体类型 | 英文名 |
|----------|--------|
| 国籍     | CONT    |
| 教育背景     | EDU    |
| 籍贯     | LOC    |
| 人名   | NAME    |
| 组织名   | ORG   |
| 专业   |  PRO  |
| 民族 | RACE   |
| 职称   | TITLE   |

## 数据评估及结果
模型在Resume测试数据评估结果:

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Resume | 94.01 | 94.36 | 94.18 |

各个类型的性能如下: 

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
|NAME | 0.9911 | 0.9911 | 0.9911|
|PRO | 0.8571 | 0.9091 | 0.8824|
|EDU | 0.9649 | 0.9821 | 0.9735|
|TITLE | 0.9405 | 0.9417 | 0.9411|
|ORG | 0.9242 | 0.9259 | 0.9250|
|CONT | 1.0000 | 1.0000|1.0000|
|RACE | 1.0000 | 1.0000 | 1.0000|
|LOC | 1.0000 | 1.0000 | 1.0000|
