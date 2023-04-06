
# 基于structbert的中文完形填空模型介绍

structbert中文完形填空模型是使用wikipedia数据和masked language model任务训练的中文自然语言理解预训练模型。

## 模型描述

我们通过引入语言结构信息的方式，将BERT扩展为了一个新模型--StructBERT。我们通过引入两个辅助任务来让模型学习字级别的顺序信息和句子级别的顺序信息，从而更好的建模语言结构。详见论文[StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding](https://arxiv.org/abs/1908.04577)
![StructBERT](./resources/StructBERT_maas.png)

## 期望模型使用方式以及适用范围
本模型主要用于生成完形填空的结果。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope-lib之后即可使用nlp_structbert_fill-mask_chinese-large的能力

#### 代码范例
```python
from modelscope.pipelines import pipeline                  
from modelscope.utils.constant import Tasks                

fill_mask_zh = pipeline(Tasks.fill_mask, model='damo/nlp_structbert_fill-mask_chinese-large')
result_zh = fill_mask_zh('段誉轻[MASK]折扇，摇了摇[MASK]，[MASK]道：“你师父是你的[MASK][MASK]，你师父可不是[MASK]的师父。你师父差得动你，你师父可[MASK]不动我。')

print(result_zh['text'])
```

### 模型局限性以及可能的偏差
模型训练数据有限，效果可能存在一定偏差。

## 训练数据介绍
数据来源于[https://huggingface.co/datasets/wikipedia](https://huggingface.co/datasets/wikipedia)

## 模型训练流程
在中文wiki等无监督数据上，通过MLM以及"模型描述"章节介绍的两个辅助任务训练了约300B字得到。
### 预处理
暂无
### 训练
暂无
## 数据评估及结果
暂无

### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用我们的文章：
```BibTex
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```
