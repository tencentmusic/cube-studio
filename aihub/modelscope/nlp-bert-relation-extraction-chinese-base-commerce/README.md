
# 商品评价解析模型介绍

商品评价解析模型是在hfl/chinese-roberta-wwm-ext预训练模型的基础上，结合自研DIRECT模型，用10%商品评价解析数据集(训练集11.7w，验证集1.3w)训练出来的关系抽取模型。

![DIRECT模型结构](modelstruct.jpg)

DIRECT为达摩院自研模型，通过使用邻接表解决了实体嵌套问题，并优化了空间复杂度，此外通过“自适应多任务学习策略”提高了模型训练效果

## 模型描述

模型基于hfl/chinese-roberta-wwm-ext，在商品评价解析数据集上通过DIRECT模型架构fine-tune得到。

## 期望模型使用方式以及适用范围
你可以使用该模型，对商品评价领域的文本进行关系抽取。
输入自然语言文本数据，模型会给出形如（属性词-情感词-情感极性）的三元组列表，支持的情感极性类型关系包括：正向情感，负向情感，中性情感

### 如何使用
在安装完成ModelScope-lib之后即可使用

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.information_extraction, 'damo/nlp_bert_relation-extraction_chinese-base-commerce', model_revision='v1.0.0')
semantic_cls(input='#纯棉白色已经收到。质量非常好。一般的厚实。尺寸规格。和商家描述的一样。') # "#"表示属性词缺省

```
### 模型局限性以及可能的偏差
模型训练数据有限，在特定行业数据上，效果可能存在一定偏差。

## 数据评估及结果

Micro-F1: 0.792

### 相关论文以及引用信息

```bib
@inproceedings{Zhao2021AdjacencyLO,
  title={Adjacency List Oriented Relational Fact Extraction via Adaptive Multi-task Learning},
  author={Fubang Zhao and Zhuoren Jiang and Yangyang Kang and Changlong Sun and Xiaozhong Liu},
  booktitle={FINDINGS},
  year={2021}
}
```