
# FAQ问答任务介绍
FAQ问答是智能对话系统(特别是垂直领域对话系统)的核心业务场景，业务专家基于经验或数据挖掘的结果，将用户会频繁问到的业务知识以Q&A的形式维护起来，称之为知识库，
当用户使用对话系统时，提问一个业务方面的问题，机器自动从知识库中找到最合适的回答。机器找答案的过程通常包含**检索**和**排序**两个阶段，针对FAQ检索和排序任务，
FAQ问答模型以structbert-base-chinese为基础，使用简单的原型网络，通过海量数据预训练(亿级)、微调(百万级)，在几个公开的数据集上都取得了不错的效果。

此外，FAQ问答任务也可以看作是一种小样本学习任务，给定每个类少量的样本(supportset)，输出query的正确的类别，因此本模型也可以用于通用的小样本分类场景；


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
本模型以structbert-base-chinese预训练模型为底座，模型结构使用简单的原型网络（如下图所示），通过小样本meta-learning的训练方式，在海量数据上进行预训练(亿级)、微调(百万级)
而得，适用于FAQ问答任务、小样本分类任务、相似度计算任务；

<div align=center><img src="./model_structure.jpg" /></div>

## 期望模型使用方式及适用范围

### 如何使用

在安装完成ModelScope之后即可使用nlp_structbert_faq-question-answering_chinese-base的的FAQ问答能力


#### 代码范例

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# query_set: query
# support_set: faq候选列表，一般实际应用场景中通过检索得到
pipeline = pipeline(Tasks.faq_question_answering, 'damo/nlp_structbert_faq-question-answering_chinese-base')
outputs = pipeline({"query_set": ["如何使用优惠券", "在哪里领券", "购物评级怎么看"],
                   "support_set": [{"text": "卖品代金券怎么用", "label": "2931733"}, 
                                   {"text": "怎么使用优惠券", "label": "2931733"}, 
                                   {"text": "这个可以一起领吗", "label": "3626004"}, 
                                   {"text": "付款时送的优惠券哪里领", "label": "3626004"}, 
                                   {"text": "购物等级怎么长", "label": "6344909"}, 
                                   {"text": "购物等级二心", "label": "6344909"}]})
# 如果输入数据中每个label只有一个句子，则可以做句子相似度计算任务

# outputs
# 输出每一个类的分值，并进行排序
# {'output': [[{'label': '6527856', 'score': 0.9982811212539673}, {'label': '1000012000', 'score': 0.0280130784958601}, {'label': '13421097', 'score': 8.978261757874861e-05}], 
#            [{'label': '1000012000', 'score': 0.8750997185707092}, {'label': '6527856', 'score': 0.0031510782428085804}, {'label': '13421097', 'score': 0.0007711253711022437}], 
#            [{'label': '13421097', 'score': 0.6274582743644714}, {'label': '1000012000', 'score': 0.0035026895347982645}, {'label': '6527856', 'score': 0.001228355336934328}]]}
```
备注：我们不限定query_set 和 support_set大小，使用者需要基于基于显存大小以及对性能的要求输入合适大小的候选集；
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 获取句子向量表示，可用于构建向量索引；
pipeline = pipeline(Tasks.faq_question_answering, 'damo/nlp_structbert_faq-question-answering_chinese-base')
sentence_vecs = pipeline.get_sentence_embedding(['如何使用优惠券', '今天有免费的10元无门槛吗', '购物评级怎么看'], max_len=30)
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

model_id = 'damo/nlp_structbert_faq-question-answering_chinese-base'
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
| Ours | 89.9 | 87.6 | 78.8 | 88.1 | 68.0 | 82.5 


```
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```