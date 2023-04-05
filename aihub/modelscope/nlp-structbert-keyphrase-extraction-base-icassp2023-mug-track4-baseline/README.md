
# ICASSP2023 MUG Challenge Track4 关键词抽取Baseline

## 赛事及背景介绍
随着数字化经济的进一步发展，越来越多的企业开始将现代信息网络作为数据资源的主要载体，并通过网络通信技术进行数据传输。同时，疫情也促使越来越多行业逐步将互联网作为主要的信息交流和分享的方式。以往的研究表明，会议记录的口语语言处理（SLP）技术如关键词提取和摘要，对于信息的提取、组织和排序至关重要，可以显著提高用户对重要信息的掌握效率。

本项目源自于ICASSP2023信号处理大挑战的通用会议理解及生成挑战赛（MUG challenge），赛事构建并发布了目前为止规模最大的中文会议数据集，并基于会议人工转写结果进行了多项SLP任务的标注；目标是推动SLP在会议文本处理场景的研究并应对其中的多项关键挑战，包括 人人交互场景下多样化的口语现象、会议场景下的长篇章文档建模 等。

## 模型介绍
针对MUG挑战赛的赛道 - 关键词抽取 任务，本模型基于[AdaSeq](https://github.com/modelscope/AdaSeq/tree/master/examples/ICASSP2023_MUG_track4)框架进行训练，基于达摩院自研预训练模型structbert结合CRF解码结构构建了对应基线。

赛道报名页面：

[https://modelscope.cn/competition/18/summary - 关键词抽取](https://modelscope.cn/competition/18/summary)

基线模型训练及推理：

https://github.com/alibaba-damo-academy/SpokenNLP

## 如何使用
关键词抽取任务可复用named-entity-recognition(命名实体识别) pipeline，
在安装ModelScope完成之后即可使用关键词抽取的能力, 默认单句长度不超过512。

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ner_pipeline = pipeline(Tasks.named_entity_recognition, 'damo/nlp_structbert_keyphrase-extraction_base-icassp2023-mug-track4-baseline')
result = ner_pipeline('哎大家好啊欢迎大家准时来参加我们的会议，今天我们会议的主题呢是如何提升我们这个洗发水儿的品牌影响力啊。我们现在是主要的产品是这个霸王防脱洗发水儿，现在请大家发表自己的意见啊欢迎大家努力的发表，请先从这位男士开始。')

print(result)
# {'output': [{'type': 'KEY', 'start': 39, 'end': 42, 'span': '洗发水'}, {'type': 'KEY', 'start': 44, 'end': 46, 'span': '品牌'}, {'type': 'KEY', 'start': 46, 'end': 49, 'span': '影响力'}, {'type': 'KEY', 'start': 59, 'end': 61, 'span': '产品'}, {'type': 'KEY', 'start': 68, 'end': 71, 'span': '洗发水'}]}
```

## 数据评估及结果
| Split |  Model   |               Backbone                | Exact/Partial F1 @10 | Exact/Partial F1 @15 | Exact/Partial F1 @20 |
|:-----:|:--------:|:-------------------------------------:|:--------------------:|:--------------------:|:--------------------:|
|  Dev  |   yake   |                   -                   |      15.0/24.3       |      19.8/30.4       |      20.4/32.1       |
|  Dev  | Bert-CRF |         sijunhe/nezha-cn-base         |      35.6/43.2       |      38.1/49.5       |      37.2/48.1       |
|  Dev  | Bert-CRF | damo/nlp_structbert_backbone_base_std |      35.9/47.7       |      40.1/52.2       |      39.4/51.1       |
