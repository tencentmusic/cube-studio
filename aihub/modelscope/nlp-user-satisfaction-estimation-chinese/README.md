
# HiTransUSE中文用户满意度估计模型介绍

用户满意度估计（User Satisfaction Estimation），又称服务满意度分析（Service Satisfaction Analysis），是分析用户是否对在线服务感到满意的重要任务。通常为输入一段连读的对话，从用户视角评价该段对话的满意度极性（不满意/中立/满意）。该任务对提升对话服务质量有重要意义，与情感分析类似，可用于各种在线客服场景（如电商客服、外呼场景等）。

## News

1. 论文：被AAAI 2023 会议录用。
	- Kaisong Song, Yangyang Kang, Jiawei Liu, Xurui Li, Changlong Sun and Xiaozhong Liu: A Speaker Turn-Aware Multi-Task Adversarial Network for Joint User Satisfaction Estimation and Sentiment Analysis..

## 模型描述

1. 模型介绍
	- （见下图左模块）使用Hierarchical Transformer Eoncoder（HiTrans）建模对话中的每条语句，包括基于bert-base-chinese的backbone建模语句对（一轮人机交互），以及基于Transformer Encoder建模上下文。
	- （见下图右模块）随后，衔接论文的Basic多任务模型，对话级的用户满意度估计（USE）为主任务，语句级的情感分析（SA）为辅助任务，两个任务联合学习互为补充。
2.  模型特点
	- 相较于原论文的BILSTM，Utterance Encoder替换为表现更好的HiTrans。
	- Basic模型使用多任务交互层来兼顾利用共享特征以及任务相关特征。
	- Enhanced模型额外考虑了角色变换信息+多任务对抗机制，通过牺牲通用性，来换取性能的进一步提升（未发布，详见原文）。
3. 模型架构

![模型结构](model.jpg)

## 期望模型使用方式以及适用范围

### 使用方式：
直接推理，输入对话直接进行推理。模型训练暂未支持，后续接入。

### 使用范围：
对话长度不超过30轮，若超过则截取最后30轮。

### 如何使用
在ModelScope框架上，提供输入的多轮对话内容，通过简单的Pipeline调用来使用满意度分析模型。

注意：一轮对话用“<b>|||</b>”连接用户提问和代理的回复语句，例如“<b>有适合我穿的尺码吗|||稍等，我看下哈</b>”；多轮对话用元组表示，例如3轮对话为<b>('Q1|||A1', 'Q2|||A2', 'Q3|||A3')</b>；对于非人机交互的一问一答场景，允许Q或A为空，即“<b>Q|||</b>”或“<b>A|||</b>”。

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_user-satisfaction-estimation_chinese', model_revision='v1.0.3')
print(semantic_cls([('返修退换货咨询|||', '手机有质量问题怎么办|||稍等，我看下', '开不开机了|||', '说话|||谢谢哈')]))
```

### 模型局限性以及可能的偏差
1. 推理模型在阿里客服两个数据集（服装和美妆）上得到的结果，理论上性能好于使用单个数据集训练。
2. 暂不支持模型训练。

## 训练数据介绍
1. Makeup：在美妆主题下的阿里客服（中文）对话数据集。
2. Clothes：在服装主题下的阿里客服（中文）对话数据集。
3.  下载地址见论文。

## 数据评估及结果
推理模型性能：满意度分析模型基于两个中文的客服对话数据集（Clothes+Makeup）训练，得到满意度最优推理结果。

| 数据集   | #dissatisfied | #neutral | #satisfied | Macro-F1 |
| -------- | -------- | -------- | --------- | ---------- |
| Clothes+Makeup |  3,482  | 7,579  | 2,479    | 0.8571      |


论文实验对比：在单独的实验数据集上的Macro-F1效果比较。

| 对比方法   | 类型 | Macro-F1 (Clothes)| Macro-F1 (Makeup)|
| -------- | -------- | -------- | --------- |
| Bert+LSTM| 单任务  | 67.90  | 74.80    |
| MILNET| 单任务  | 63.81  | 75.30   |
| CAMIL| 单任务  | 70.40 | 78.60   |
| MT-ES| 多任务  | 68.04  | 76.06    |
| Meta-LSTM| 多任务  | 69.05  | 76.65    |
| RSSN| 多任务  | 69.51 | 79.18  |
| Ours (Enhanced)| 多任务  | 71.11  | 80.11    |
		


## 论文引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的论文：

```bib
@inproceedings{conf/aaai/SongKLLSL23,
  author    = {Kaisong Song and Yangyang Kang and Jiawei Liu and Xurui Li and Changlong Sun and Xiaozhong Liu},
  title     = {A Speaker Turn-Aware Multi-Task Adversarial Network for Joint User Satisfaction Estimation and Sentiment Analysis},
  booktitle = {Proceedings of the AAAI 2023},
  pages     = {198--207},
  year      = {2019}
}
```

#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/damo/nlp_user-satisfaction-estimation_chinese.git
```