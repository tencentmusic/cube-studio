
# 印尼语电商域Title NER介绍

## 模型描述
本方法采用Transformer-CRF模型，使用XLM-RoBERTa作为预训练模型底座。本模型主要用于给输入越南语商品标题文本产出命名实体识别结果, 具体调用方式请参考代码示例。

## 训练数据介绍
- ecom-title-id: 内部印尼语电商领域商品标题命名实体识别(NER)数据集, 支持产品(product), 功能(function), 品牌(brand), 模式(pattern), 颜色(color), 用户群体(consumer_group), 风格(style)等七大类型的实体识别

| 实体类型 | 英文名 |
|----------|------|
| 产品 | product |
| 功能 | function |
| 品牌 | brand |
| 图案 | pattern |
| 颜色 | color |
| 用户群体 | consumer_group |
| 风格 | style |

## 快速上手
### 适用范围
在安装ModelScope完成之后即可使用named-entity-recognition(命名实体识别)的能力, 默认单句长度不超过512, 推荐输入长度不超过128的句子。

### 代码示例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_xlmr_named-entity-recognition_indo-ecommerce-title', model_revision='v1.0.1')
result = ner_pipeline('Bayi Bayi Anak Musim Dingin Hangat Tebal Faux Bulu')

print(result)
#{'output': [{'type': 'consumer_group', 'start': 0, 'end': 4, 'span': 'Bayi'}, {'type': 'consumer_group', 'start': 5, 'end': 9, 'span': 'Bayi'}, {'type': 'consumer_group', 'start': 10, 'end': 14, 'span': 'Anak'}]}
```

## 性能评测

### 全局评测
| Precision | Recall | F1 |
| --- | --- | --- |
| 86.5 | 85.1 | 85.8 |


### 按实体类型评测
| 实体类型 | Precision | Recall | F1 |
| --- | --- | --- | --- |
| product | 85.85 | 87.04 | 86.44 |
| function | 81.65 | 63.11 | 71.20 |
| brand | 78.63 | 75.21 | 76.88 |
| pattern | 80.87 | 74.71 | 77.67 |
| color | 83.28 | 88.05 | 85.60 |
| consumer_group | 96.30 | 96.52 | 96.41|
| style | 93.73 | 92.42 | 93.07 |