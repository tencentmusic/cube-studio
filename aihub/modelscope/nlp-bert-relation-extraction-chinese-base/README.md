
# 百科关系抽取模型介绍

百科关系抽取模型是在hfl/chinese-roberta-wwm-ext预训练模型的基础上，用duie数据集训练出来的关系抽取模型。

## 模型描述

模型基于hfl/chinese-roberta-wwm-ext，在duie数据集上fine-tune得到。

## 期望模型使用方式以及适用范围
你可以使用该模型，对通用领域的文本进行关系抽取。
输入自然语言文本数据，模型会给出形如（主语，谓语，宾语）的三元组列表，支持的关系包括：毕业院校、嘉宾、配音、主题曲、代言人、所属专辑、父亲、作者、上映时间、母亲、专业代码、占地面积、邮政编码、票房、注册资本、主角、妻子、编剧、气候、歌手、获奖、校长、创始人、首都、丈夫、朝代、饰演、面积、总部地点、祖籍、人口数量、制片人、修业年限、所在城市、董事长、作词、改编自、出品公司、导演、作曲、主演、主持人、成立日期、简称、海拔、号、国籍、官方语言。 

### 如何使用
在安装完成ModelScope-lib之后即可使用

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.information_extraction, 'damo/nlp_bert_relation-extraction_chinese-base')
semantic_cls(input='高捷，祖籍江苏，本科毕业于东南大学')

```

### 模型局限性以及可能的偏差
模型训练数据有限，在特定行业数据上，效果可能存在一定偏差。

## 训练数据介绍
数据来源于https://aistudio.baidu.com/aistudio/competition/detail/46


## 数据评估及结果

Micro-F1: 0.761

### 相关论文以及引用信息

```bib
@inproceedings{Zhao2021AdjacencyLO,
  title={Adjacency List Oriented Relational Fact Extraction via Adaptive Multi-task Learning},
  author={Fubang Zhao and Zhuoren Jiang and Yangyang Kang and Changlong Sun and Xiaozhong Liu},
  booktitle={FINDINGS},
  year={2021}
}
```