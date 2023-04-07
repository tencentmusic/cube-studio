
# Bert实体向量-中文-通用领域-base

实体链接旨在区分文本中的mention和大规模知识图谱中实体的对应关系，也是自然语言处理(NLP)领域的基础问题，在很多对话、检索、关系抽取等下游任务中发挥着重要作用。通常来说，实体链接分为两个步骤：召回和排序。这里我们主要关注于实体召回阶段。近些年来，基于预训练语言模型的实体表示模型在公开数据集和实际运用中都表现出了极佳的水平。因此，我们也着重关注于实体表征模型的构建。

对于具体的实体链接任务，我们需要输入一个句子和对应的mention的位置，最终模型会反馈对应的kb中的实体结果。

## 双塔实体召回模型

基于监督数据训练的实体表示模型通常采用双塔模型。如下图所示，在框架中，通过对于mention和实体分别编码，我们采用预训练语言模型[CLS]位置的向量作为最终的向量表示。最后通过点积距离作为二者之间的相关性的度量标准。

<div align=center><img width="450" height="380" src="./resources/el_model.png" /></div>

## 使用方式和范围

使用方式:
- 直接推理, 对于给定的mention和实体获得其对应的向量表示，计算对应的相似度

使用范围:
- 本模型主要用于实体链接任务，需要预先提供ner的结果

### 如何使用

在ModelScope框架上，提供输入文本(默认最长文本长度为128)，将需要计算向量表征的实体，使用[ENT_S]和[ENT_E]标签进行标记，即可以通过简单的Pipeline调用来使用实体向量模型。
ModelScope封装了统一的接口对外提供实体向量表示和相似度计算方法。

#### 代码示例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipeline_ee = pipeline(Tasks.sentence_embedding, "damo/nlp_bert_entity-embedding_chinese-base")

inputs = {
        "source_sentence": ["宋小宝小品《美人鱼》， [ENT_S] 大鹏 [ENT_E] 上演生死离别，关键时刻美人鱼登场"],
        "sentences_to_compare": [
            "董成鹏； 类型： Person； 别名： Da Peng， 大鹏;",
            "超级飞侠； 类型： Work； 别名： 超飞， 출동!슈퍼윙스， Super Wings;",
            "王源； 类型： Person； 别名： Roy;",
        ]
    }


result = pipeline_ee(input=inputs)
print(result)
```

### 训练流程

- 模型: 双塔实体表示模型, 采用双塔模型作为模型基座
- 二阶段训练: 模型训练分为两阶段, 一阶段的负样本数据只有in-batch的随机负样本, 二阶段可以从一阶段得到的困难负样本采样得到

模型采用2张NVIDIA V100机器训练, 超参设置如下:
```
train_epochs=3
max_sequence_length=128
batch_size=32
learning_rate=2e-5
optimizer=AdamW
```

### 模型效果评估

我们主要在实体链接场景下评估模型效果, 在CCKS 2020短文本实体链接任务上召回评估结果如下:

| Model | Recall@4 | Recall@16 | Recall@128 | 
|:-----:|:--------:|:---------:|:----------:|
| Bert  |  80.24   |   80.24   |   99.89    |

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