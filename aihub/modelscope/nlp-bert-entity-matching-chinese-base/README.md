
# Bert实体相关性-中文-通用领域-base

实体链接旨在区分文本中的mention和大规模知识图谱中实体的对应关系，也是自然语言处理(NLP)领域的基础问题，在很多对话、检索、关系抽取等下游任务中发挥着重要作用。通常来说，实体链接分为两个步骤：召回和排序。这里我们主要关注于实体排序阶段，这和之前的语义相关性方法非常相似。近些年来，基于预训练语言模型的NLP模型在公开数据集和实际运用中都表现出了极佳的水平。

对于具体的实体链接任务，我们需要输入一个句子和对应的mention的位置，最终模型会反馈对应的庞大的知识库中的实体结果。通常需要首先从知识库中召回较小的候选文档集合，然后再对这些候选文档进行更加复杂的排序，产出最后的排序结果。本模型也是基于预训练的排序模型。

## 实体排序模型

基于监督数据训练的实体排序模型通常采用单塔模型。如下图右侧所示，在框架中，通过将所有候选项依次拼接到输入文本中，我们可以使用预训练语言模型[CLS]位置的向量用于得到最终的排序分数。

<div align=center><img width="800" height="380" src="./resources/ce_model.png" /></div>

## 使用方式和范围

使用方式:
- 直接推理, 对于给定的mention和实体获得其对应的相似度

使用范围:
- 本模型主要用于实体链接任务的排序阶段，需要预先提供ner的结果和一阶段召回的实体候选项

### 如何使用

在ModelScope框架上，提供输入文本(默认最长文本长度为128)，其中源句子中的实体需要用特殊标签[ENT_S]和[ENT_E]进行标记，即可以通过简单的Pipeline调用来使用实体向量模型。ModelScope封装了统一的接口对外提供实体向量表示和相似度计算方法。

#### 代码示例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipeline_em = pipeline(task=Tasks.text_ranking, model='damo/nlp_bert_entity-matching_chinese-base')

inputs = {
        "source_sentence": ["我是猫》([日]夏目漱石)【摘要 [ENT_S] 书评 [ENT_E]  试读】"],
        "sentences_to_compare": [
            "书评； 类型： Other； 别名： Book review; 三元组: 书评 # 外文名 # Book review $ 书评 # 摘要 # 书评，即评论并介绍书籍的文章，是以“书”为对象，实事求是的、有见识的分析书籍的形式和内容，探求创作的思想性、学术性、知识性和艺术性，从而在作者、读者和出版商之间构建信息交流的渠道。 $ 书评 # 定义 # 评论并介绍书籍的文章 $ 书评 # 中文名 # 书评 $ 书评 # 义项描述 # 书评 $ 书评 # 类型 # 应用写作的一种重要文体 $ 书评 # 标签 # 文学作品、文化、出版物、小说、书籍 $ ', '书评； 类型： Other； 别名： Book review; 三元组: 书评 # 外文名 # Book review $ 书评 # 摘要 # 书评，即评论并介绍书籍的文章，是以“书”为对象，实事求是的、有见识的分析书籍的形式和内容，探求创作的思想性、学术性、知识性和艺术性，从而在作者、读者和出版商之间构建信息交流的渠道。 $ 书评 # 定义 # 评论并介绍书籍的文章 $ 书评 # 中文名 # 书评 $ 书评 # 义项描述 # 书评 $ 书评 # 类型 # 应用写作的一种重要文体 $ 书评 # 标签 # 文学作品、文化、出版物、小说、书籍 $",
            "摘要； 类型： Other； 别名： 摘， abstract， 书评; 三元组: 摘要 # 读音 # zhāi yào $ 摘要 # 外文名 # abstract $ 摘要 # 摘要 # 摘要又称概要、内容提要，意思是摘录要点或摘录下来的要点。 $  摘要 # 词目 # 摘要 $ 摘要 # 词性 # 动词，名词 $ 摘要 # 中文名 # 摘要 $ 摘要 # 别称 # 概要、内容提要 $ 摘要 # 义项描述 # 摘要 $ 摘要 # 标签 # 文化、文学家、行业人物、法律术语、小说 $",
            "歌手； 类型： Person； 别名： singer， 男歌手， 女歌手; 三元组: 歌手 # 外文名 # singer $ 歌手 # 摘要 # 歌手，泛指演唱歌曲及其他声乐作品的娱乐业人士，也用于自称，作为职业它有一个规范的叫法“歌唱演员”。 $ 歌手 # 三大流派 # 美声、民族、流行 $ 歌手 # 唱腔分类 # 男声，女声 $ 歌手 # 中文名 # 歌手 $ 歌手 # 别称 # 歌唱 演员 $ 歌手 # 义项描述 # 职业名称 $ 歌手 # 释义 # 演唱歌曲及其他声乐作品的娱乐业人士 $ 歌手 # 一般分类 # 唱片歌手，网络歌手，选秀歌手 $ 歌手 # 标签 # 文化术语、电影、书籍、音乐 $",
        ]
    }


result = pipeline_em(input=inputs)
print(result)
```

### 训练流程

- 模型: 实体排序模型, 采用单塔模型作为模型基座
- 训练: 模型使用动态采样技术从一阶段召回的负样本池中采样得到困难负样本和一定的随机负样本

模型采用4张NVIDIA V100机器训练, 超参设置如下:
```
train_epochs=5
max_sequence_length=128
batch_size=32
learning_rate=2e-5
optimizer=AdamW
```

### 模型效果评估

我们主要在实体链接场景下评估模型效果, 在CCKS 2020短文本实体链接任务上召回评估结果如下:

| Model |  F1   | 
|:-----:|:-----:|
| Bert  | 87.98 |

## 引用

```BibTeX
@article{Huang2022KENER,
  title={{DAMO-NLP} at {NLPCC-2022} Task 2: Knowledge Enhanced Robust {NER}},
  author={Shen Huang and Yuchen Zhai and Xinwei Long and Yong Jiang and Xiaobin Wang and Yin Zhang and Pengjun Xie},
  series={Lecture Notes in Computer Science},
  volume={13552},
  pages={284--293},
  publisher={Springer},
  year={2022},
  url={https://doi.org/10.1007/978-3-031-17189-5\_24},
  doi={10.1007/978-3-031-17189-5\_24}
}

@inproceedings{ma-etal-2021-muver,
    title = "{M}u{VER}: {I}mproving First-Stage Entity Retrieval with Multi-View Entity Representations",
    author = "Ma, Xinyin  and
      Jiang, Yong  and
      Bach, Nguyen  and
      Wang, Tao  and
      Huang, Zhongqiang  and
      Huang, Fei  and
      Lu, Weiming",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.205",
    doi = "10.18653/v1/2021.emnlp-main.205",
    pages = "2617--2624",
    abstract = "Entity retrieval, which aims at disambiguating mentions to canonical entities from massive KBs, is essential for many tasks in natural language processing. Recent progress in entity retrieval shows that the dual-encoder structure is a powerful and efficient framework to nominate candidates if entities are only identified by descriptions. However, they ignore the property that meanings of entity mentions diverge in different contexts and are related to various portions of descriptions, which are treated equally in previous works. In this work, we propose Multi-View Entity Representations (MuVER), a novel approach for entity retrieval that constructs multi-view representations for entity descriptions and approximates the optimal view for mentions via a heuristic searching method. Our method achieves the state-of-the-art performance on ZESHEL and improves the quality of candidates on three standard Entity Linking datasets.",
}
```