
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

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_lstm_named-entity-recognition_chinese-social_media')
result = ner_pipeline('叶赟葆：全球时尚财运滚滚而来钱')

print(result)
# {'output': [{'type': 'PER.NAM', 'start': 0, 'end': 3, 'span': '叶赟葆'}]}
```


### 模型局限性以及可能的偏差
本模型基于weibo数据集(社交媒体领域)上训练，在垂类领域中文文本上的NER效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
- [Weibo](https://doi.org/10.18653/v1/D15-1064) 社交媒体领域中文命名实体识别公开数据集，包括地缘政治实体、地名、机构名、人名及其代指，共1889个句子。

| 实体类型 | 英文名 |
|----------|--------|
|  地缘政治实体名   | GPE.NAM    |
| 地缘政治指代   | GPE.NOM   |
| 地名   |  LOC.NAM  |
| 地名指代   |  LOC.NOM  |
| 机构名   |  ORG.NAM  |
| 机构名指代   |  ORG.NOM  |
| 人名   |  PER.NAM  |
| 人名指代   |  PER.NOM  |


## 数据评估及结果
模型在weibo测试数据评估结果:

| Dataset | Precision | Recall | F1 |
| --- | --- | --- | --- |
| weibo | 66.45 | 47.09 | 55.12 |

各个类型的性能如下:

|Dataset | Precision | Recall |F1|
|---|---|---|---|
|PER.NAM | 72.50 | 48.33 | 58.00|
|GPE.NAM | 57.14 | 59.57 | 58.33|
|ORG.NOM | 85.71 | 35.29 | 50.00|
|PER.NOM | 74.05 | 55.11 | 63.19|
|LOC.NAM | 50.00 | 15.79 | 24.00|
|LOC.NOM | 33.33 | 44.44 | 38.10|
|GPE.NOM | 00.00 | 00.00 | 00.00|
|ORG.NAM | 31.58 | 15.38 | 20.69|


