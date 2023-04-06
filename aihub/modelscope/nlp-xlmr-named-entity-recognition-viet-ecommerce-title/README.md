
# 越南语电商域Title NER介绍

## 模型描述
本方法采用Transformer-CRF模型，使用XLM-RoBERTa作为预训练模型底座。本模型主要用于给输入越南语商品标题文本产出命名实体识别结果, 具体调用方式请参考代码示例。

## 训练数据介绍
- ecom-title-th: 内部越南语电商领域商品标题命名实体识别(NER)数据集, 支持产品(product), 功能(function), 品牌(brand), 模式(pattern), 颜色(color), 用户群体(consumer_group), 风格(style)等七大类型的实体识别

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

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_xlmr_named-entity-recognition_viet-ecommerce-title', model_revision='v1.0.1')
result = ner_pipeline('Nón vành dễ thương cho bé gái')

print(result)
{'output': [{'type': 'product', 'start': 0, 'end': 8, 'span': 'Nón vành'}, {'type': 'style', 'start': 9, 'end': 18, 'span': 'dễ thương'}, {'type': 'consumer_group', 'start': 23, 'end': 29, 'span': 'bé gái'}]}
```

## 性能评测

### 全局评测
| Precision | Recall | F1 |
| --- | --- | --- |
| 80.5 | 75.2 | 77.8 |


### 按实体类型评测
| 实体类型 | Precision | Recall | F1 |
| --- | --- | --- | --- |
| product | 81.10 | 82.00 | 81.55 |
| brand | 77.34 | 80.54 | 78.91 |
| pattern | 66.48 | 71.40 | 68.85 |
| color | 86.73 | 91.26 | 88.93 |
| consumer_group | 94.55 | 95.85 | 95.19 |
| style | 84.76 | 90.26 | 87.42 |