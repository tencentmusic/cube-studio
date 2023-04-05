

# StructBERT中文事实准确性检测模型介绍
事实准确性检测（fact-checking）任务指判断一对句子对（句子1，句子2）在语义上是否存在事实冲突(包括三类：事实一致、事实冲突、不确定)，可以看作是自然语言推理（NLI）任务的一种类型。
StructBERT事实准确性检测是在structbert-base-chinese预训练模型的基础上，使用24万fact-checking业务数据训练出来的自然语言推理模型。（具体样例可参考代码示例）

## 模型描述

StructBERT中文事实准确性检测模型基于Structbert-base-chinese，按照BERT论文中的方式，在fact-checking业务数据上fine-tune得到。

![模型结构](model.jpg)

## 期望模型使用方式以及适用范围

你可以使用StructBERT中文事实准确性检测模型，对电商领域事实准确性检测任务进行推理。
输入形如（句子1，句子2）的句子对数据，模型会给出该句子对应的自然语言推理标签 {"事实一致": 0, "事实冲突": 1, "不确定": 2} 以及相应的概率。

### 如何使用
在安装完成ModelScope-lib，请参考  [modelscope环境安装](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85) 。


#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.nli, 'damo/nlp_structbert_fact-checking_chinese-base',model_revision='v1.0.1')
semantic_cls(input=('好通透呀', '亲，店铺隐形眼镜分为透/:809明片和彩片。★透明片以矫正视力为主，透明片有无色透明或淡蓝色透明（淡蓝色主要是方便摘戴，配戴基本看不出颜色）。'))
semantic_cls(input=('适合老人用', '【70直径 加大款】适合体重小于200斤【80直径 特大款】适合体重小于240斤'))

```
#### 微调代码范例
- 敬请期待


#### 模型局限性以及可能的偏差
模型训练数据有限，不能包含所有行业，因此在特定行业数据上，效果可能存在一定偏差。

## 训练数据介绍

训练数据来源于业务数据，暂不公开。




### 相关论文以及引用信息

```bib
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```