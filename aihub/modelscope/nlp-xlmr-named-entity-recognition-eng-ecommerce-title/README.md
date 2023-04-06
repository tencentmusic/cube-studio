
# 英语电商域Title NER介绍

## 模型描述
本方法采用Transformer-CRF模型，使用XLM-RoBERTa作为预训练模型底座。本模型主要用于给输入英语商品标题文本产出命名实体识别结果, 具体调用方式请参考代码示例。

## 训练数据介绍
- ecom-title-en: 内部英语电商领域商品标题命名实体识别(NER)数据集, 支持产品(product), 功能(function), 品牌(brand), 模式(pattern), 颜色(color), 用户群体(consumer_group), 风格(style)等七大类型的实体识别

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

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_xlmr_named-entity-recognition_eng-ecommerce-title', model_revision='v1.0.1')
result = ner_pipeline('T2P Bicycle Phone Mount Universal Phone Mount Holder Stand Cradle Clamp')

print(result)
#{'output': [{'type': 'product', 'start': 4, 'end': 23, 'span': 'Bicycle Phone Mount'}, {'type': 'function', 'start': 24, 'end': 33, 'span': 'Universal'}, {'type': 'product', 'start': 34, 'end': 52, 'span': 'Phone Mount Holder'}, {'type': 'product', 'start': 59, 'end': 71, 'span': 'Cradle Clamp'}]}
```

## 性能评测

### 全局评测
| Precision | Recall | F1 |
| --- | --- | --- |
| 74.6 | 80.2 | 77.3 |


### 按实体类型评测
| 实体类型 | Precision | Recall | F1 |
| --- | --- | --- | --- |
| product | 72.96 | 77.55 | 75.18 |
| function | 69.84 | 78.76 | 74.03 |
| brand | 87.77 | 90.86 | 89.29 |
| pattern | 68.07 | 71.58 | 69.78 |
| color | 83.39 | 86.89 | 85.10 |
| consumer_group | 94.77 | 95.97 | 95.36 |
| style | 85.79 | 82.04 | 83.87 |