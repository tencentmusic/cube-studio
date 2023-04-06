
# StructBERT中文情绪分类模型介绍

情绪分类任务，通常为输入一段句子或一段话，识别该句话情绪类别的模型。 在用户评价、观点抽取、意图识别中往往起到重要作用。

## 模型描述

模型基于Structbert-base-chinese，在情绪分类的数据集上fine-tune得到。 模型可识别的情绪包含（恐惧、愤怒、厌恶、喜好、悲伤、高兴、惊讶）七种类别。

![模型结构](model.jpg)

## 期望模型使用方式以及适用范围

你可以使用StructBERT中文情绪分类模型，对通用领域的中文情绪分类任务进行推理。 输入自然语言文本，模型会给出该文本的情绪分类标签以及相应的概率。

### 如何使用

在安装完成ModelScope之后即可使用

#### 推理代码范例

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_structbert_emotion-classification_chinese-base')
semantic_cls(input='新年快乐！')
```

### 模型局限性以及可能的偏差

模型训练数据有限，效果可能存在一定偏差。

## 训练数据介绍

使用数据集分布情况如下：

|   | 高兴   | 悲伤   | 厌恶   | 喜好 | 恐惧  | 惊讶   | 愤怒 | 无明显情绪 |
|-------|-------|-------|------|----------|------|------|------|-------|
| 训练集 | 10610 | 13418 | 6909  | 8236 | 820 | 1664 | 5312 | 32071 |
| 验证集 | 1326 | 1677 | 863  | 1029 | 102 | 207 | 663 | 4008 |

## 数据评估及结果
 
该模型在验证集上的f1为0.5743。

## 数据评估及结果
```bib
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```