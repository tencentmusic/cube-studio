
# ICASSP2023 MUG Challenge Track2 抽取式篇章摘要Baseline

## 赛事及背景介绍
随着数字化经济的进一步发展，越来越多的企业开始将现代信息网络作为数据资源的主要载体，并通过网络通信技术进行数据传输。同时，疫情也促使越来越多行业逐步将互联网作为主要的信息交流和分享的方式。
以往的研究表明，会议记录的口语语言处理（SLP）技术如关键词提取和摘要，对于信息的提取、组织和排序至关重要，可以显著提高用户对重要信息的掌握效率。
本项目源自于ICASSP2023信号处理大挑战的通用会议理解及生成挑战赛（MUG challenge），赛事构建并发布了目前为止规模最大的中文会议数据集，并基于会议人工转写结果进行了多项SLP任务的标注；
目标是推动SLP在会议文本处理场景的研究并应对其中的多项关键挑战，包括 人人交互场景下多样化的口语现象、会议场景下的长篇章文档建模等。

## 模型介绍
针对MUG挑战赛的赛道-抽取式摘要-抽取式篇章摘要任务，我们使用阿里巴巴达摩院自研模型[PoNet](https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary)构建了对应基线。

PoNet是一种具有线性复杂度(O(N))的序列建模模型，使用pooling网络替代Transformer模型中的self-attention来进行上下文的建模。
PoNet模型主要由三个不同粒度的pooling网络组成，一个是全局的pooling模块(GA)，分段的segment max-pooling模块(SMP)，和局部的max-pooling模块(LMP)，对应捕捉不同粒度的序列信息：
1.	在第一阶段，GA沿着序列长度进行平均得到句子的全局表征g。为了加强对全局信息的捕捉，GA在第二阶段对g和输入训练计算cross-attention。由于g的长度为1，因此总的计算复杂度仍为O(N)。
2.	SMP按每个分段求取最大值，以捕获中等颗粒度的信息。
3.	LMP沿着序列长度的方向计算滑动窗口max-pooling。
4.	然后通过池化融合(PF)将这些池化特征聚合起来。由于GA的特征在整个token序列是共享的，SMP的特征在segment内部也是共享的，直接将这些特征加到原始token上会使得token趋同（向量加法），而这种token表征同质化的影响将会降低诸如句子对分类任务的性能。
因此，我们在PF层将原始的token于对应的GA，SMP特征计算元素乘法得到新的特征，使得不同的token对应了不同的特征。

针对抽取式篇章摘要，segment是采用的句子粒度，对篇章进行抽取式摘要。

赛道报名页面：https://modelscope.cn/competition/13/summary


## 使用方式
直接输入篇章文本，输出若干句摘要。

## 模型局限性以及可能的偏差
模型采用AliMeeting4MUG Corpus语料进行训练，在其他领域文本上的抽取式篇章摘要性能可能会有偏差。

## 训练方式
模型用[nlp_ponet_fill-mask_chinese-base](https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary)初始化，在AliMeeting4MUG Corpus的抽取式篇章摘要训练数据上进行训练。
初始学习率为5e-5，batch_size为2，max_seq_length=4096

More Details: https://github.com/alibaba-damo-academy/SpokenNLP


## 模型效果评估
在MUG的Session-level Extractive Summarization 的开发集结果如下，我们报告了基于三个参考的平均和最佳 Rouge-1,2,L 分数。

| Model      | Backbone                                                                                                            | Ave. R1 | Ave. R2 | Ave. RL | Max R1 | Max R2 | Max RL | 
|------------|---------------------------------------------------------------------------------------------------------------------|---------|---------|---------|--------|--------|--------|  
| Longformer | IDEA-CCNL/Erlangshen-Longformer-110M                                                                                | 56.17   | 	29.52  | 38.20   | 61.75  | 36.84  | 47.06  |
| PoNet      | [damo/nlp_ponet_fill-mask_chinese-base](https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary) | 57.08   | 29.90   | 38.36   | 62.20  | 37.34  | 46.98  |

注：Longformer是五个不同种子的平均结果。


## 代码范例
```python

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(
    task=Tasks.extractive_summarization,
    model='damo/nlp_ponet_extractive-summarization_doc-level_chinese-base')

result = p(documents='移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。')

print(result[OutputKeys.TEXT])
```

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

# Baseline of ICASSP2023 MUG Challenge Track2 Session-level Extractive Summarization

## Competition and background introduction
Meetings are vital for seeking information, sharing knowledge, and improving productivity. Technological advancements as well as the pandemic rapidly increased amounts of meetings. Prior studies show that spoken language processing (SLP) technologies on meeting transcripts, such as keyphrase extraction and summarization, are crucial for distilling, organizing, and prioritizing information and significantly improve users' efficiency in grasping salient information.
This project originated from the ICASSP2023 Signal Processing Grand Challenge - General Meeting Understanding and Generation (MUG) challenge. The event built and released the largest Chinese conference data set so far, and conducted a number of labeling of SLP tasks based on manual meeting transcripts.
The goal is to promote the research of SLP in meeting transcripts and address several key challenges, including diverse spoken language phenomena in human interaction scenarios, long-form document modeling in meeting scenarios, etc.

## Model description
For the Track2 Session-level Extractive Summarization of the MUG Challenge, we used the self-developed model of Alibaba DAMO Academy [PoNet] (https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary) to construct a baseline .

PoNet is a Pooling Network (PoNet) for token mixing in long sequences with linear complexity, which uses a pooling network to replace the self-attention in the Transformer model for context modeling.
The PoNet model is mainly composed of three pooling networks with different granularities, one is the global pooling module (GA), the segment max-pooling module (SMP), and the local max-pooling module (LMP), corresponding to capturing different granularities of sequence. 
1. In the first stage, GA averages along the sequence length to obtain the global representation g of the sentence. In order to strengthen the capture of global information, GA calculates cross-attention for g and input training in the second stage. Since the length of g is 1, the total computational complexity is still O(N).
2. SMP is maximized per segment to capture moderately granular information.
3. LMP calculates the sliding window max-pooling along the direction of the sequence length.
4. These pooled features are then aggregated by Pooling Fusion (PF). Since the features of GA are shared throughout the token sequence, the features of SMP are also shared within the segment. Directly adding these features to the original token will make the token converge (vector addition), and the homogeneity of token representation will degrade performance on tasks such as sentence pair classification.
Therefore, we multiplied the original token with the corresponding GA and SMP feature calculation elements at the PF layer to obtain new features, so that different tokens correspond to different features.

For the session-level extractive summarization task, the sentence is denoted as segment granularity. 

Track registration page: https://modelscope.cn/competition/12/summary

## How to use
Input the session-level text and output several sentence summaries.

## Model limitations and possible bias
The model is trained with the AliMeeting4MUG Corpus, and the performance of extractive topic summarization on texts in other fields may be biased.

## Training method
The model is initialized with [nlp_ponet_fill-mask_chinese-base](https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary), and trained on the extractive topic summary training data of AliMeeting4MUG Corpus.
The initial learning rate is 5e-5, batch size is 2, max_seq_length=4096

More Details: https://github.com/alibaba-damo-academy/SpokenNLP

## Evaluation
The development set results of Session-level Extractive Summarization at MUG are as follows, and we report the average and best Rouge-1,2,L scores based on three references.

| Model      | Backbone                                                                                                            | Ave. R1 | Ave. R2 | Ave. RL | Max R1 | Max R2 | Max RL | 
|------------|---------------------------------------------------------------------------------------------------------------------|---------|---------|---------|--------|--------|--------|  
| Longformer | IDEA-CCNL/Erlangshen-Longformer-110M                                                                                | 56.17   | 	29.52  | 38.20   | 61.75  | 36.84  | 47.06  |
| PoNet      | [damo/nlp_ponet_fill-mask_chinese-base](https://modelscope.cn/models/damo/nlp_ponet_fill-mask_chinese-base/summary) | 57.08   | 29.90   | 38.36   | 62.20  | 37.34  | 46.98  |

Note: Longformer is the average result of five different seeds.

## Code example
```python

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(
    task=Tasks.extractive_summarization,
    model='damo/nlp_ponet_extractive-summarization_doc-level_chinese-base')

result = p(documents='移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。')

print(result[OutputKeys.TEXT])
```

### Related work and citation information
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

