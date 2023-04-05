
# PoNet完形填空模型-中文-base介绍

本模型选用PoNet模型结构，通过Masked Language Modeling（MLM）和Sentence Structural Objective（SSO）预训练任务在中文Wikipedia数据预训练获得，可以用于完形填空任务，也可以作为初始化模型在下游自然语言理解任务上finetune来使用。

## 模型描述
[PoNet](https://arxiv.org/abs/2110.02442)是一种具有线性复杂度(O(N))的计算模型，使用pooling网络替代Transformer模型中的self-attention来对句子词汇进行混合，具体包括在local、segment、global三个粒度上的pooling网络，从而捕捉上下文信息。
其结构如下图所示。

![PoNet](./resources/PoNet.png)

实验表明，PoNet在长文本测试Long Range Arena(LRA)榜上在准确率上比Transformer高2.28个点，在GPU上运行速度是Transformer的9倍，显存占用只有1/10。此外，实验也展示了PoNet的迁移学习能力，PoNet-Base在GLUE基准上达到了BERT-Base的95.7%的准确性。
详见论文[PoNet: Pooling Network for Efficient Token Mixing in Long Sequences](https://arxiv.org/abs/2110.02442)

## 期望模型使用方式以及适用范围
本模型主要用于生成完形填空的结果。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope-lib之后即可使用nlp_ponet_fill-mask_chinese-base的能力

### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipeline_ins = pipeline(Tasks.fill_mask, model='damo/nlp_ponet_fill-mask_chinese-base')
input = "人民文学出版社[MASK]1952[MASK]，出版《[MASK][MASK]演义》、《[MASK]游记》、《水浒传》、《[MASK]楼梦》，合为“[MASK]大名著”。"
print(pipeline_ins(input))
```

### 模型局限性以及可能的偏差
- 模型训练数据有限，效果可能存在一定偏差。
- 当前版本在pytorch 1.11和pytorch 1.12环境测试通过，其他环境可用性待测试

## 训练数据介绍

数据来源于[https://dumps.wikimedia.org/](https://dumps.wikimedia.org/)

## 模型训练流程
在中文Wikipedia的无监督数据上，通过MLM和SSO任务训练得到。

### 预处理
对于训练数据会采用如下预处理，对于MLM任务，掩蔽概率设置为15%。80%的掩蔽位置被[MASK]替换，10%被替换为随机抽样的单词，剩下的10%不变。对于SSO任务，包含多个段落的序列在随机位置被截断为两个子序列，其中
1/3概率用另一个随机选择的子序列替换其中一个子序列，1/3的概率交换两个子序列，1/3的概率不变。

### 训练细节
在中文Wikipedia上使用Adam优化器，初始学习率为1e-4，batch_size为384。

## 数据评估及结果
在下游任务finetune后，[CAIL](https://cail.oss-cn-qingdao.aliyuncs.com/cail2019/CAIL2019-SCM.zip)、[CLUE](https://huggingface.co/datasets/clue)的开发集结果如下：

| Dataset  | CAIL  | AFQMC | CMNLI | CSL   | IFLYTEK | OCNLI | TNEWS | WSC   |
|----------|-------|-------|-------|-------|---------|-------|-------|-------|
| Accuracy | 61.93 | 70.25 | 72.9  | 72.97 | 58.21   | 68.14 | 55.04 | 64.47 |

在下游任务[MUG](https://modelscope.cn/datasets/modelscope/Alimeeting4MUG/summary)的Topic Segmentation 和 Topic-level and Session-level Extractive Summarization的开发集结果如下：

| Task                                                                                                                     | Positive F1 | 
|--------------------------------------------------------------------------------------------------------------------------|-------------|
| [Topic Segmentation](https://modelscope.cn/models/damo/nlp_ponet_document-segmentation_topic-level_chinese-base/summary) | 0.251       |

| Task                                                                                                                    | Ave. R1 | Ave. R2 | Ave. RL | Max R1 | Max R2 | Max RL | 
|-------------------------------------------------------------------------------------------------------------------------|---------|---------|---------|--------|--------|--------|  
| [Session-Level ES](https://modelscope.cn/models/damo/nlp_ponet_extractive-summarization_doc-level_chinese-base/summary) | 57.08   | 29.90   | 38.36   | 62.20  | 37.34  | 46.98  |
| [Topic-Level ES](https://modelscope.cn/models/damo/nlp_ponet_extractive-summarization_topic-level_chinese-base/summary) | 52.86   | 35.80   | 46.09   | 66.67  | 54.05  | 63.14  | 

More Details: https://github.com/alibaba-damo-academy/SpokenNLP

## 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用我们的文章：
```BibTex
@inproceedings{DBLP:journals/corr/abs-2110-02442,
  author    = {Chao{-}Hong Tan and
               Qian Chen and
               Wen Wang and
               Qinglin Zhang and
               Siqi Zheng and
               Zhen{-}Hua Ling},
  title     = {{PoNet}: Pooling Network for Efficient Token Mixing in Long Sequences},
  booktitle = {10th International Conference on Learning Representations, {ICLR} 2022,
               Virtual Event, April 25-29, 2022},
  publisher = {OpenReview.net},
  year      = {2022},
  url       = {https://openreview.net/forum?id=9jInD9JjicF},
}
```

# Introduction

This model uses the PoNet structure, which is pre-trained on Chinese Wikipedia data through Masked Language Modeling (MLM) and Sentence Structural Objective (SSO) pre-training tasks. It can be used for cloze tasks, and can also be used as an initialization model for downstream natural language understanding.

## Model description
[PoNet](https://arxiv.org/abs/2110.02442) is a pooling network with linear complexity (O(N)), which uses pooling network instead of self-attention in Transformer model to mix tokens.
It uses multi-granularity pooling and pooling fusion to capture different levels of contextual information and combine their interactions with tokens.
The structure is shown in the figure below.

![PoNet](./resources/PoNet.png)

## Expected model usage and scope of application
This model is mainly used to generate cloze results. Users can try various input documents by themselves. Please refer to the code example for the specific calling method.

### How to use
After installing ModelScope-lib, you can use the ability of nlp_ponet_fill-mask_chinese-base.

### Code example
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipeline_ins = pipeline(Tasks.fill_mask, model='damo/nlp_ponet_fill-mask_chinese-base')
input = "人民文学出版社[MASK]1952[MASK]，出版《[MASK][MASK]演义》、《[MASK]游记》、《水浒传》、《[MASK]楼梦》，合为“[MASK]大名著”。"
print(pipeline_ins(input))
```

### Model limitations and possible bias
- The model training data is limited, and the effect may have certain deviations.
- The current version has passed the test in pytorch 1.11 and pytorch 1.12 environments, and the usability of other environments is yet to be tested.

## Training data introduction

The data comes from [https://dumps.wikimedia.org/](https://dumps.wikimedia.org/)

## Model training
On the unsupervised data of Chinese Wikipedia, it is trained by MLM and SSO tasks.

### Preprocessing
For the training data, the following preprocessing is used. For the MLM task, the masking probability is set to 15%. 80% of the masked positions are replaced by [MASK], 10% are replaced by randomly sampled words, and the remaining 10% are unchanged. 
For the SSO task, a long sequence containing several paragraphs is truncated into two subsequences at random positions, with 1/3 probability of replacing one of the subsequences with another randomly selected subsequence, 1/3 probability of swapping the two subsequences, and 1/3 probability unchanged. These three cases are assigned three different labels for the ternary classification.

### Training detail
Using the Adam optimizer on Chinese Wikipedia, the initial learning rate is 1e-4, and the batch size is 384.

## Data evaluation and results
After the downstream task finetune, the development set results of [CAIL](https://cail.oss-cn-qingdao.aliyuncs.com/cail2019/CAIL2019-SCM.zip) and [CLUE](https://huggingface.co/datasets/clue) are as follows:

| Dataset  | CAIL  | AFQMC | CMNLI | CSL   | IFLYTEK | OCNLI | TNEWS | WSC   |
|----------|-------|-------|-------|-------|---------|-------|-------|-------|
| Accuracy | 61.93 | 70.25 | 72.9  | 72.97 | 58.21   | 68.14 | 55.04 | 64.47 |

The development set results of Topic Segmentation and Topic-level and Session-level Extractive Summarization in the downstream task MUG are as follows:

| Task                                                                                                                     | Positive F1 | 
|--------------------------------------------------------------------------------------------------------------------------|-------------|
| [Topic Segmentation](https://modelscope.cn/models/damo/nlp_ponet_document-segmentation_topic-level_chinese-base/summary) | 0.251       |

| Task                                                                                                                    | Ave. R1 | Ave. R2 | Ave. RL | Max R1 | Max R2 | Max RL | 
|-------------------------------------------------------------------------------------------------------------------------|---------|---------|---------|--------|--------|--------|  
| [Session-Level ES](https://modelscope.cn/models/damo/nlp_ponet_extractive-summarization_doc-level_chinese-base/summary) | 57.08   | 29.90   | 38.36   | 62.20  | 37.34  | 46.98  |
| [Topic-Level ES](https://modelscope.cn/models/damo/nlp_ponet_extractive-summarization_topic-level_chinese-base/summary) | 52.86   | 35.80   | 46.09   | 66.67  | 54.05  | 63.14  | 

More Details: https://github.com/alibaba-damo-academy/SpokenNLP


## Related work and citation information
If our model is helpful to you, please cite our paper:
```BibTex
@inproceedings{DBLP:journals/corr/abs-2110-02442,
  author    = {Chao{-}Hong Tan and
               Qian Chen and
               Wen Wang and
               Qinglin Zhang and
               Siqi Zheng and
               Zhen{-}Hua Ling},
  title     = {{PoNet}: Pooling Network for Efficient Token Mixing in Long Sequences},
  booktitle = {10th International Conference on Learning Representations, {ICLR} 2022,
               Virtual Event, April 25-29, 2022},
  publisher = {OpenReview.net},
  year      = {2022},
  url       = {https://openreview.net/forum?id=9jInD9JjicF},
}
```