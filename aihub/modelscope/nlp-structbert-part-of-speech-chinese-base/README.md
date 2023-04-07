
# BAStructBERT通用领域中文词性标注模型介绍

词性标注任务是将给定句子中的每个单词从给定标签组 (tag set)中赋予一个词性标签 (part-of-speech tag)。中文词性标注任务示例:

- 输入: 中文词性标注模型
- 输出: 中文 NN 词性 NN 标注 NN 模型 NN

## 模型描述

中文词性标注可以采用分词后的词序列作为输入, 模型输出每个词的词性标签。基于预训练语言模型的词性标注模型通常采用联合分词+词性标注的序列标注模型。这里我们基于通用新闻领域CTB9标注数据训练模型, 采用无监督统计特征增强的StructBERT+softmax序列标注模型,序列标注标签体系(B、I、E、S),四个标签分别表示标签处于该单词的起始、中间、终止位置或者改单词独立成词; 以StructBERT预训练语言模型为底座的序列标注模型可以参考下面的模型图:

<div align=center><img width="450" height="300" src="resources/pos_model.png" /></div>

StructBERT预训练语言模型可以参考 [StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding](https://arxiv.org/abs/1908.04577)。为了进一步提升中文词性标注模型的效果，在StructBERT模型基础上, 通过在预训练过程中增加大规模无监督词汇边界统计信息可以有效提升预训练模型对词汇边界的识别能力。我们实验验证融合词汇边界信息的预训练模型Boundary Aware StructBERT (BAStructBERT)模型在绝大多数中文序列标注任务上有进一步的效果提升。BAStructBERT模型结构和基础的StructBERT模型一致, BAStructBERT模型的预训练流程示意图如下所示, 更加详细的模型结构和实验结果将在后续公开的论文中介绍。

<div align=center><img src="./resources/bastructbert.png" /></div>


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
# Version less than 1.1 please use TokenClassificationPreprocessor
from modelscope.preprocessors import TokenClassificationTransformersPreprocessor

pipeline_ins = pipeline(task=Tasks.part_of_speech)
result = pipeline_ins(input="今天天气不错，适合出去游玩")
print (result)
# {'output': [{'type': 'NT', 'start': 0, 'end': 2, 'span': '今天'}, {'type': 'NN', 'start': 2, 'end': 4, 'span': '天气'}, {'type': 'VA', 'start': 4, 'end': 6, 'span': '不错'}, {'type': 'PU', 'start': 6, 'end': 7, 'span': '，'}, {'type': 'VV', 'start': 7, 'end': 9, 'span': '适合'}, {'type': 'VV', 'start': 9, 'end': 11, 'span': '出去'}, {'type': 'VV', 'start': 11, 'end': 13, 'span': '游玩'}]}
model_id = 'damo/nlp_structbert_part-of-speech_chinese-base'
model = Model.from_pretrained(model_id)
tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
pipeline_ins = pipeline(task=Tasks.token_classification, model=model, preprocessor=tokenizer)
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
CTB6数据集标注数据样例:
```
上海_NR 二月_NT 十日_NT 电_NN （_PU 记者_NN 谢金虎_NR 、_PU 张持坚_NR ）_PU
```

数据预处理成(B、I、E、S)标签体系的数据格式, 每一个独立的单字对应一个独立的标签, 预处理后数据样例如下:
```
上 海 二 月 十 日 电 （ 记 者 谢 金 虎 、 张 持 坚 ）	B-NR E-NR B-NT E-NT B-NT E-NT S-NN S-PU B-NN E-NN B-NR I-NR E-NR S-PU B-NR I-NR E-NR S-PU
```

### 训练
模型采用2张NVIDIA V100机器训练, 超参设置如下:

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
| 95.04 | 95.16 | 94.91 |

### 模型训练示例代码
如果需要基于自己的数据对分词模型进行二次训练, 建议可以采用ModelScope提供的序列标注理解框架**AdaSeq**进行模型训练, **AdaSeq**是一个基于ModelScope的一站式NLP序列理解开源工具箱，支持高效训练自定义模型，旨在提高开发者和研究者们的开发和创新效率，助力模型快速定制和前沿论文工作>落地。

1. 安装AdaSeq

```shell
pip install adaseq
```

2. 模型微调

准备训练配置，将下面的代码保存为train.yaml。

该配置中的数据集为示例数据集[CTB6中文pos训练数据](https://modelscope.cn/datasets/dingkun/chinese_pos_ctb6/summary)，如需使用自定义数据或调整参数，可参考《[AdaSeq模型训练最佳实践](https://github.com/modelscope/AdaSeq/blob/master/docs/tutorials/training_a_model_zh.md)》，准备数据或修改配置文件。AdaSeq中也提供了大量的[模型、论文、比赛复现示例]([https://github.com/modelscope/AdaSeq/tree/master/examples](https://github.com/modelscope/AdaSeq/tree/master/examples))，欢迎大家使用。``yaml``文件示例如下:

```yaml
experiment:
  exp_dir: experiments/
  exp_name: ctb6_pos
  seed: 42

task: word-segmentation

dataset:
  data_file:
    train: https://modelscope.cn/api/v1/datasets/dingkun/chinese_pos_ctb6/repo?Revision=master&FilePath=train.txt
    dev: https://modelscope.cn/api/v1/datasets/dingkun/chinese_pos_ctb6/repo?Revision=master&FilePath=dev.txt
    test: https://modelscope.cn/api/v1/datasets/dingkun/chinese_pos_ctb6/repo?Revision=master&FilePath=test.txt
  data_type: conll

preprocessor:
  type: sequence-labeling-preprocessor
  max_length: 256
  tag_scheme: BIES

data_collator: SequenceLabelingDataCollatorWithPadding

model:
  type: sequence-labeling-model
  embedder:
    model_name_or_path: damo/nlp_structbert_part-of-speech_chinese-base
  dropout: 0.1
  use_crf: true

train:
  max_epochs: 1
  dataloader:
    batch_size_per_gpu: 32
  optimizer:
    type: AdamW
    lr: 2.0e-5
    param_groups:
      - regex: crf
        lr: 2.0e-1
  lr_scheduler:
    type: LinearLR
    start_factor: 1.0
    end_factor: 0.0
    total_iters: 30

evaluation:
  dataloader:
    batch_size_per_gpu: 64
  metrics:
    - type: ner-metric
    - type: ner-dumper
      model_type: sequence_labeling
      dump_format: conll
```

运行命令开始训练。在GPU上训练需要至少6G显存，可以根据实际GPU情况调整batch_size等参数。

```shell
adaseq train -c train.yaml
```

3. 模型文件

二进制模型文件和相关配置文件会保存在 `./experiments/ctb6_pos/${yymmddHHMMSS.ffffff}/output/`

4. 模型推理
需要指出的是, 上面的示例``yaml``配置中采用的crf解码方式, 所以最后训练得到分词模型是BERT-crf结构而非BERT-softmax结果(实测两者的效果很接近)。在推理阶段, 为了能做BERT-crf结构的推理, 我们可以采用ModelScope内置的针对NER任务的pipeline进行推理, 示例代码如下:

```python
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
# pipeline = pipeline(Tasks.named_entity_recognition, ${model_save_path})
pipeline = pipeline(Tasks.named_entity_recognition, "./experiments/ctb6_pos/${yymmddHHMMSS.ffffff}/output/")
pipeline('美好世界')
# 输出结果如下:
# {'output': [{'type': 'JJ', 'start': 0, 'end': 2, 'span': '美好'}, {'type': 'NN', 'start': 2, 'end': 4, 'span': '世界'}]}
```

## 引用
StructBERT模型可以参考论文
```BibTex
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```
