
# 基于bert的中文完形填空模型介绍
基于bert的中文完形填空模型是用wikipedia_zh/baike/news训练的中文自然语言理解预训练模型。

## 模型描述

本模型是对于BERT模型的中文版本进行复现优化的版本。除了采用更多的数据和更多的迭代轮次之外，训练细节尽可能忠实原文。BERT原论文参见[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## 期望模型使用方式以及适用范围
本模型主要用于生成完形填空的结果。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope之后即可使用nlp_bert_fill-mask_chinese-base的能力

#### 代码范例
```python
from modelscope.pipelines import pipeline                  
from modelscope.utils.constant import Tasks                

fill_mask_zh = pipeline(Tasks.fill_mask, model='damo/nlp_bert_fill-mask_chinese-base')
result_zh = fill_mask_zh('段誉轻[MASK]折扇，摇了摇[MASK]，[MASK]道：“你师父是你的[MASK][MASK]，你师父可不是[MASK]的师父。你师父差得动你，你师父可[MASK]不动我。')                 

print(result_zh['text'])
```

### 模型局限性以及可能的偏差
模型在数据集上训练，有可能产生一些偏差，请用户自行评测后决定如何使用。

## 训练数据介绍
数据来源于[https://huggingface.co/datasets/wikipedia](https://huggingface.co/datasets/wikipedia)

## 模型训练流程
在中文wiki等无监督数据上，通过MLM和NSP任务训练了约300B字得到
### 预处理
暂无
### 训练
暂无
## 数据评估及结果
暂无
### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用文章：
```BibTex
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```