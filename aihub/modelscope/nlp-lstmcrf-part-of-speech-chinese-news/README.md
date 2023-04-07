
# LSTM通用领域中文词性标注模型介绍

词性标注任务是将给定句子中的每个单词从给定标签组 (tag set)中赋予一个词性标签 (part-of-speech tag)。中文词性标注任务示例:

- 输入: 中文词性标注模型
- 输出: 中文 NN 词性 NN 标注 NN 模型 NN

## 模型描述

本方法采用char-BiLSTM-CRF模型，word-embedding使用[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)。char-BiLSTM-CRF模型具体结构可以参考论文[Neural Architectures for Named Entity Recognition](https://aclanthology.org/N16-1030.pdf)

[CTB](https://verbs.colorado.edu/chinese/ctb.html)标注数据采用的标签体系:

| 标签 |                  描述                 |                  含义                  | 标签 |                 描述                |          含义          |
|:----:|:-------------------------------------:|:--------------------------------------:|:----:|:-----------------------------------:|:----------------------:|
|  AD  |                adverbs                |                  副词                  |   M  | Measure word(including classifiers) |    量词，例子：“个”    |
|  AS  |              Aspect marke             | 体态词，体标记（例如：了，在，着，过） |  MSP |            Some particles           |       例子：“所”       |
|  BA  |             把 in ba-const            |          “把”、“将”的词性标记          |  NN  |             Common nouns            |        普通名词        |
|  CC  |        Coordinating conjunction       |             并列连词，“和”             |  NR  |             Proper nouns            |        专有名词        |
|  CD  |            Cardinal numbers           |              数字，“一百”              |  NT  |            Temporal nouns           | 时序词，表示时间的名词 |
|  CS  |           Subordinating conj          |     从属连词（例子：若，如果，如…）    |  OD  |           Ordinal numbers           |     序数词，“第一”     |
|  DEC |       的 for relative-clause etc      |              “的”词性标记              |  ON  |             Onomatopoeia            |     拟声词，“哈哈”     |
|  DEG |              Associative              |               联结词“的”               |   P  |  Preposition (excluding 把 and 被)  |          介词          |
|  DER |    in V-de construction, and V-de-R   |                  “得”                  |  PN  |               pronouns              |          代词          |
|  DEV |               before VP               |                   地                   |  PU  |             Punctuations            |          标点          |
|  DT  |               Determiner              |              限定词，“这”              |  SB  |       in long bei-construction      |     例子：“被，给”     |
|  ETC | Tag for words, in coordination phrase |                等，等等                |  SP  |       Sentence-final particle       |    句尾小品词，“吗”    |
|  FW  |             Foreign words             |                例子：ISO               |  VA  |        Predicative adjective        |    表语形容词，“红”    |
|  IJ  |              interjetion              |                 感叹词                 |  VC  |                Copula               |      系动词，“是”      |
|  JJ  |     Noun-modifier other than nouns    |                                        |  VE  |         有 as the main verb         |          “有”          |
|  LB  |        in long bei-construction       |              例子：被，给              |  VV  |             Other verbs             |        其他动词        |
|  LC  |               Localizer               |           定位词，例子：“里”           |      |                                     |                        |

## 期望模型使用方式以及适用范围
本模型主要用于给输入中文句子的分词以及词性标注结果。用户可以自行尝试输入中文句子。具体调用方式请参考代码示例。

### 如何使用
在安装ModelScope完成之后即可使用chinese-part-of-speech(中文词性标注)的能力, 默认单句长度不超过512。

#### 代码范例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.preprocessors import TokenClassificationTransformersPreprocessor

model_id = 'damo/nlp_lstmcrf_part-of-speech_chinese-news'
model = Model.from_pretrained(model_id)
tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
pipeline_ins = pipeline(Tasks.part_of_speech, model=model, preprocessor=tokenizer)
result = pipeline_ins(input="今天天气不错，适合出去游玩")
print (result)
#{'output': [{'type': 'NT', 'start': 0, 'end': 2, 'span': '今天'}, {'type': 'NN', 'start': 2, 'end': 4, 'span': '天气'}, {'type': 'VA', 'start': 4, 'end': 6, 'span': '不错'}, {'type': 'PU', 'start': 6, 'end': 7, 'span': '，'}, {'type': 'VV', 'start': 7, 'end': 9, 'span': '适合'}, {'type': 'VV', 'start': 9, 'end': 11, 'span': '出去'}, {'type': 'VV', 'start': 11, 'end': 13, 'span': '游玩'}]}
```

### 模型局限性以及可能的偏差
本模型基于CTB9数据集(通用新闻领域)上训练，在垂类领域中文文本上的中文词性标注效果会有降低，请用户自行评测后决定如何使用。

## 训练数据介绍
本模型采用新闻领域词性标注数据集CTB9标注训练。

## 模型训练流程

### 预处理
CTB9数据集标注数据样例:
```
上海_NR 浦东_NR 开发_NN 与_CC 法制_NN 建设_NN 同步_VV
新华社_NN 上海_NR 二月_NT 十日_NT 电_NN （_PU 记者_NN 谢金虎_NR 、_PU 张持坚_NR ）_PU
上海_NR 浦东_NR 近年_NT 来_LC 颁布_VV 实行_VV 了_AS 涉及_VV 经济_NN 、_PU 贸易_NN 、_PU 建设_NN 、_PU 规划_NN 、_PU 科技_NN 、_PU 文教_NN 等_ETC 领域_NN 的_DEC 七十一_CD 件_M 法规性_NN 文件_NN ，_PU 确保_VV 了_AS 浦东_NR 开发_NN 的_DEG 有序_JJ 进行_NN 。_PU
```

数据预处理成(B、I、E、S)标签体系的数据格式, 每一个独立的单字对应一个独立的标签, 预处理后数据样例如下:
```
上 海 浦 东 开 发 与 法 制 建 设 同 步	B-NR E-NR B-NR E-NR B-NN E-NN S-CC B-NN E-NN B-NN E-NN B-VV E-VV
新 华 社 上 海 二 月 十 日 电 （ 记 者 谢 金 虎 、 张 持 坚 ）	B-NN I-NN E-NN B-NR E-NR B-NT E-NT B-NT E-NT S-NN S-PU B-NN E-NN B-NR I-NR E-NR S-PU B-NR I-NR E-NR S-PU
上 海 浦 东 近 年 来 颁 布 实 行 了 涉 及 经 济 、 贸 易 、 建 设 、 规 划 、 科 技 、 文 教 等 领 域 的 七 十 一 件 法 规 性 文 件 ， 确 保 了 浦 东 开 发 的 有 序 进 行 。 	B-NR E-NR B-NR E-NR B-NT E-NT S-LC B-VV E-VV B-VV E-VV S-AS B-VV E-VV B-NN E-NN S-PU B-NN E-NN S-PU B-NN E-NN S-PU B-NN E-NN S-PU B-NN E-NN S-PU B-NN E-NN S-ETC B-NN E-NN S-DEC B-CD I-CD E-CD S-M B-NN I-NN E-NN B-NN E-NN S-PU B-VV E-VV S-AS B-NR E-NR B-NN E-NN S-DEG B-JJ E-JJ B-NN E-NN S-PU
```

### 训练
模型采用1张NVIDIA V100机器训练, 超参设置如下:

```
train_epochs=10
max_sequence_length=256
batch_size=64
learning_rate=5e-5
optimizer=AdamW
```

### 数据评估及结果

模型在CTB不同年份的测试数据评估结果(F1 score):

| CTB5 | CTB6 | CTB9    |
|:-----------:|:--------:|:-------:|
| 90.08 | 90.16 | 90.47 |

## 论文引用
char-BiLSTM-CRF模型可以参考下列论文
```BibTex
@inproceedings{lample-etal-2016-neural,
    title = "Neural Architectures for Named Entity Recognition",
    author = "Lample, Guillaume  and
      Ballesteros, Miguel  and
      Subramanian, Sandeep  and
      Kawakami, Kazuya  and
      Dyer, Chris",
    booktitle = "Proceedings of the 2016 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N16-1030",
    doi = "10.18653/v1/N16-1030",
    pages = "260--270",
}
``` 
