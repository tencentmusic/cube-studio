
# FAQ问答任务介绍

FAQ问答是智能对话系统(特别是垂直领域对话系统)的核心业务场景，业务专家基于经验或数据挖掘的结果，将用户会频繁问到的业务知识以Q&A的形式维护起来，称之为知识库，
当用户使用对话系统时，提问一个业务方面的问题，机器自动从知识库中找到最合适的回答。机器找答案的过程通常包含**检索**和**排序**两个阶段;

**MGIMN FAQ问答模型**以StructBERT预训练模型为底座，模型结构采用多粒度交互式匹配小样本模型(参考[NAACL2022论文](https://aclanthology.org/2022.naacl-main.141/))
相对于StructBERT FAQ问答模型效果更优；

**[<font size=3>FAQ问答开发实践</font>](https://developer.aliyun.com/article/1027733?spm=a2c6h.14164896.0.0.1c1d3483goazti)**

<hr>

## 相关模型
- [StructBERT FAQ问答-中文-通用领域-base](https://www.modelscope.cn/models/damo/nlp_structbert_faq-question-answering_chinese-base/summary) 通用FAQ问答模型，适用于各个领域，效果和性能兼具，落地应用的最优选择；
- [StructBERT FAQ问答-中文-政务领域-base](https://www.modelscope.cn/models/damo/nlp_structbert_faq-question-answering_chinese-gov-base/summary) 基于StructBERT FAQ问答-中文-通用领域-base模型在政务数据上进行finetune得到，适用于政务领域FAQ问答任务，包括但不局限于社保、公积金等场景；
- [StructBERT FAQ问答-中文-金融领域-base](https://www.modelscope.cn/models/damo/nlp_structbert_faq-question-answering_chinese-finance-base/summary) 基于StructBERT FAQ问答-中文-通用领域-base模型在金融数据上进行finetune得到，适用于金融领域FAQ问答任务，包括但不局限于银行、保险等场景；
- [MGIMN FAQ问答-中文-通用领域-base](https://www.modelscope.cn/models/damo/nlp_mgimn_faq-question-answering_chinese-base/summary) 采用多粒度交互式小样本模型设计，相对StructBERT FAQ问答-中文-通用领域-base模型，效果更优，但推理速度会下降，方法参考论文：[NAACL2022](https://aclanthology.org/2022.naacl-main.141/) 
- [MGIMN FAQ问答-中文-政务领域-base](https://www.modelscope.cn/models/damo/nlp_mgimn_faq-question-answering_chinese-gov-base/summary) 基于MGIMN FAQ问答-中文-通用领域-base模型在政务数据上进行finetune得到，适用于政务领域FAQ问答任务；包括但不局限于社保、公积金等场景；
- [MGIMN FAQ问答-中文-金融领域-base](https://www.modelscope.cn/models/damo/nlp_mgimn_faq-question-answering_chinese-finance-base/summary) 基于MGIMN FAQ问答-中文-通用领域-base模型在金融数据上进行finetune得到，适用于金融领域FAQ问答任务；包括但不局限于银行、保险等场景；
- [FAQ问答-多语言-通用领域-base](https://www.modelscope.cn/models/damo/nlp_faq-question-answering_multilingual-base/summary)，多语言FAQ模型，支持英、俄、西、法、阿、韩、葡、越南、泰语、印尼语、菲律宾语、马来语、粤语等多种语言；

## 模型描述
本模型以structbert-base-chinese预训练模型为底座，模型结构采用多粒度交互式匹配小样本模型结构（如下图所示），核心的模块主要有：Instance Matching Layer和Class-wise Aggregation Layer，
详细的模型结构设计请参考NAACL2022论文[MGIMN:Multi-Grained Interactive Matching Network for Few-shot Text Classification](https://aclanthology.org/2022.naacl-main.141/)，模型通过小样本meta-learning的训练方式，在海量数据上进行预训练(亿级)、微调(百万级)；适用于
FAQ问答任务和通用小样本分类任务；（说明：模型结构与论文中一致，但训练方式由于框架约束会略有差异）

<div align=center><img src="./model_structure.jpg" /></div>

## 期望模型使用方式及适用范围

### 如何使用

在安装完成ModelScope之后即可使用damo/nlp_mgimn_faq-question-answering_chinese-base的的FAQ问答能力


#### 代码范例

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# query_set: query
# support_set: faq候选列表，一般实际应用场景中通过检索得到
faq_pipeline = pipeline(Tasks.faq_question_answering, 'damo/nlp_mgimn_faq-question-answering_chinese-base')
outputs = faq_pipeline({"query_set": ['如何使用优惠券', '在哪里领券', '购物评级怎么看'],
                   "support_set": [{'text': '卖品代金券怎么用', 'label': '2931733'}, 
                                   {'text': '怎么使用优惠券', 'label': '2931733'}, 
                                   {'text': '这个可以一起领吗', 'label': '3626004'}, 
                                   {'text': '付款时送的优惠券哪里领', 'label': '3626004'}, 
                                   {'text': '购物等级怎么长', 'label': '6344909'}, 
                                   {'text': '购物等级二心', 'label': '6344909'}]})

# outputs
# 输出每一个类的分值，并进行排序
# {'output': [[{'label': '2931733', 'score': 0.9460447430610657},
#   {'label': '3626004', 'score': 1.7818173319028574e-06},
#   {'label': '6344909', 'score': 1.9170591780692803e-09}],
#  [{'label': '3626004', 'score': 0.9911342859268188},
#   {'label': '2931733', 'score': 0.003209104062989354},
#   {'label': '6344909', 'score': 1.5378593616333092e-06}],
#  [{'label': '6344909', 'score': 0.999358594417572},
#   {'label': '3626004', 'score': 2.7110197606816655e-06},
#   {'label': '2931733', 'score': 1.836718638514867e-06}]]}

```
备注：我们不限定query_set 和 support_set大小，使用者需要基于基于显存大小以及对性能的要求输入合适大小的候选集；
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 获取句子向量表示，可用于构建向量索引；
faq_pipeline = pipeline(Tasks.faq_question_answering, 'damo/nlp_mgimn_faq-question-answering_chinese-base')
sentence_vecs = faq_pipeline.get_sentence_embedding(['如何使用优惠券', '今天有免费的10元无门槛吗', '购物评级怎么看'], max_len=30)
```
备注：同样，我们也不对输入进行大小限制，使用者需要基于显存大小输入合适大小的数据；


### 模型局限性及可能的偏差

虽然我们的数据尽量覆盖各行业，但仍有可能不适用于某些特定行业；

## 训练数据介绍

训练数据来源于业务数据

## 模型训练&Finetune
```python
import os

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.hub import read_config

model_id = 'damo/nlp_mgimn_faq-question-answering_chinese-base'
tmp_dir='./work_dir'

# 我们仅以此数据作为示例，并不能作为faq任务的训练数据
train_dataset = MsDataset.load(
    'jd', namespace='DAMO_NLP',
    split='train').remap_columns({'sentence': 'text'})
# 用户可以加载自己的数据集，格式如下，训练数据少时会报错，尽量保证label数和每个label对应样本数不少于5；
#train_dataset = [{'text':'测试数据1','label':'label1'},{'text':'测试数据3','label':'label1'},
#                 {'text':'测试数据2','label':'label2'},{'text':'测试数据4','label':'label2'},.....]
eval_dataset = MsDataset.load(
    'jd', namespace='DAMO_NLP',
    split='validation').remap_columns({'sentence': 'text'})

cfg: Config = read_config(model_id)
cfg.train.train_iters_per_epoch = 50
cfg.evaluation.val_iters_per_epoch = 2
cfg.evaluation.metrics = 'accuracy'
cfg.train.seed = 1234
cfg.train.hooks = [{
    'type': 'CheckpointHook',
    'by_epoch': False,
    'interval': 50
}, {
    'type': 'EvaluationHook',
    'by_epoch': False,
    'interval': 50
}, {
    'type': 'TextLoggerHook',
    'by_epoch': False,
    'rounding_digits': 5,
    'interval': 10
}]
cfg_file = os.path.join(tmp_dir, 'config.json')
cfg.dump(cfg_file)

trainer = build_trainer(
    Trainers.faq_question_answering_trainer,
    default_args=dict(
        model=model_id,
        work_dir=tmp_dir,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cfg_file=cfg_file))

trainer.train()

evaluate_result = trainer.evaluate()
print(evaluate_result)

```

## 数据评估及结果

由于公开的垂直领域FAQ数据较少，我们选择几个公开的意图分类数据进行数据评估，其中banking/OOS/clinic为中文翻译数据；
- 评测方式：划分训练集和测试集，对于测试集中的每一个query，从训练集中检索top10个类，每个类至多5条样本，基于检索出的样本数据对query进行分类
- 评测指标：ACC

| Model | fewjoint | nlpccst4 | banking | clinic | OOS | Average | 
| --- | --- | --- | --- | --- | --- | -- | 
| SentenceTransformer(paraphrase-multilingual-mpnet-base-v2)  | 79.1 | 81.9 | 75.7 | 86.1 | 66.1 | 77.8
| SentenceTransformer(paraphrase-multilingual-MiniLM-L12-v2) | 82.5 | 81.0 | 77.4 | 84.1 | 64.6 | 77.9 
| StructBERT FAQ问答-中文-通用领域-base | 89.9 | 87.6 | 78.8 | 88.1 | 68.0 | 82.5 
| **MGIMN FAQ问答-中文-通用领域-base(当前模型)** | 89.7 | 91.4 | 80.6 | 88.8 | 69.2 | 83.9


```
@inproceedings{zhang-etal-2022-mgimn,
    title = "{MGIMN}: Multi-Grained Interactive Matching Network for Few-shot Text Classification",
    author = "Zhang, Jianhai  and
      Maimaiti, Mieradilijiang  and
      Xing, Gao  and
      Zheng, Yuanhang  and
      Zhang, Ji",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.141",
    abstract = "Text classification struggles to generalize to unseen classes with very few labeled text instances per class.In such a few-shot learning (FSL) setting, metric-based meta-learning approaches have shown promising results. Previous studies mainly aim to derive a prototype representation for each class.However, they neglect that it is challenging-yet-unnecessary to construct a compact representation which expresses the entire meaning for each class.They also ignore the importance to capture the inter-dependency between query and the support set for few-shot text classification. To deal with these issues, we propose a meta-learning based method MGIMN which performs instance-wise comparison followed by aggregation to generate class-wise matching vectors instead of prototype learning.The key of instance-wise comparison is the interactive matching within the class-specific context and episode-specific context. Extensive experiments demonstrate that the proposed method significantly outperforms the existing SOTA approaches, under both the standard FSL and generalized FSL settings.",
}
```