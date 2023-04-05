
# 泰语通用领域分词模型介绍

## 任务介绍
泰语分词目的是将连续的泰语字符分隔成具有语言学意义的泰语单词，是泰语文本理解的基础模块。
- 输入: ...รถคันเก่าก็ยังเก็บเอาไว้ยังไม่ได้ขาย...
- 输出: .../ รถ/ คัน/ เก่า/ ก็/ ยัง/ เก็บ/ เอา/ ไว้/ ยัง/ ไม่/ ได้/ ขาย/ ...

## 模型介绍
- 本方法采用Transformer-Linear模型，使用XLM-RoBERTa(XLM-R)作为预训练模型底座。
- 对于输入文本，本模型逐字符预测当前字符是否为泰语单词边界, 具体调用方式请参考代码示例。

## 训练数据介绍
- BEST-2010

## 快速上手
### 适用范围
在安装ModelScope完成之后即可使用named-entity-recognition(命名实体识别)的能力, 默认单句包含字符数不超过512。

### 代码示例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.word_segmentation, 'damo/nlp_xlmr_word-segmentation_thai')
result = ner_pipeline('...รถคันเก่าก็ยังเก็บเอาไว้ยังไม่ได้ขาย...')

print(result)
#{'output': ['...', 'รถ', 'คัน', 'เก่า', 'ก็', 'ยัง', 'เก็บ', 'เอา', 'ไว้', 'ยัง', 'ไม่', 'ได้', 'ขาย', '...']}
```

## 性能评测
| Precision | Recall | F1 |
| --- | --- | --- |
| 97.9 | 97.9 | 97.9 |