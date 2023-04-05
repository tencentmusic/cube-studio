
# PoNet完形填空模型-英文-base介绍

本模型选用PoNet模型结构，通过Masked Language Modeling（MLM）和Sentence Structural Objective（SSO）预训练任务在英文Bookcorpus+Wikitext数据预训练获得，可以用于完形填空任务，也可以作为初始化模型在下游自然语言理解任务上finetune来使用。

## 模型描述
[PoNet](https://arxiv.org/abs/2110.02442)是一种具有线性复杂度(O(N))的计算模型，使用pooling网络替代Transformer模型中的self-attention来对句子词汇进行混合，具体包括在local、segment、global三个粒度上的pooling网络，从而捕捉上下文信息。
其结构如下图所示。

![PoNet](./resources/PoNet.png)

实验表明，PoNet在长文本测试Long Range Arena(LRA)榜上在准确率上比Transformer高2.28个点，在GPU上运行速度是Transformer的9倍，显存占用只有1/10。此外，实验也展示了PoNet的迁移学习能力，PoNet-Base在GLUE基准上达到了BERT-Base的95.7%的准确性。
详见论文[PoNet: Pooling Network for Efficient Token Mixing in Long Sequences](https://arxiv.org/abs/2110.02442)


## 期望模型使用方式以及适用范围
本模型主要用于生成完形填空的结果。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope-lib之后即可使用nlp_ponet_fill-mask_english-base的能力

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipeline_ins = pipeline(Tasks.fill_mask, model='damo/nlp_ponet_fill-mask_english-base')
input = 'Everything in [MASK] you call reality is really [MASK] a reflection of your [MASK]. Your [MASK] universe is just a mirror [MASK] of your story.'
print(pipeline_ins(input))
```

### 模型局限性以及可能的偏差
- 模型训练数据有限，效果可能存在一定偏差。
- 当前版本在pytorch 1.11和pytorch 1.12环境测试通过，其他环境可用性待测试

## 训练数据介绍
数据来源于[bookcorpus](https://huggingface.co/datasets/bookcorpus)和[wikitext](https://huggingface.co/datasets/wikitext)

## 模型训练流程
在英文Bookcorpus+Wikitext的无监督数据上，通过MLM和SSO任务训练得到。

### 预处理
对于训练数据会采用如下预处理，对于MLM任务，掩蔽概率设置为15%。80%的掩蔽位置被[MASK]替换，10%被替换为随机抽样的单词，剩下的10%不变。对于SSO任务，包含多个段落的序列在随机位置被截断为两个子序列，其中
1/3概率用另一个随机选择的子序列替换其中一个子序列，1/3的概率交换两个子序列，1/3的概率不变。

### 训练
在英文Bookcorpus+Wikitext上使用Adam优化器，初始学习率为1e-4，batch_size为192。

## 数据评估及结果
在下游任务finetune后，GLUE的开发集结果如下：

| Dataset | MNLI(m/mm) | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | AVG |
| --- | --- | --- | --- | --- | --- | --- | --- |  --- | --- |
| Accuracy | 76.99/77.21 | 87.55 | 84.33 | 89.22 | 45.36 | 84.57 | 81.76 | 64.26 | 76.80 |

长文本分类数据集结果如下：

| Dataset | HND | IMDb | Yelp-5 | Arxiv |
| --- | --- | --- | --- | --- | 
| Accuracy | 96.2 | 93.0 | 69.13 | 86.11 | 

更多结果详见论文[PoNet: Pooling Network for Efficient Token Mixing in Long Sequences](https://arxiv.org/abs/2110.02442)

### 相关论文以及引用信息
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
