

<!--- 以下model card模型说明部分，请使用中文提供（除了代码，bibtex等部分） --->

# StructBERT辱骂风险识别-中文-外呼-tiny
本模型基于StructBERT-tiny模型，使用外呼场景下的辱骂风险识别数据集训练得到。

## 模型描述

本模型是在中文预训练模型StructBERT的基础上使用外呼辱骂风险识别数据进行微调得到的。关于StructBERT的详细介绍可参见https://www.modelscope.cn/models/damo/nlp_structbert_backbone_base_std/summary 。

## 期望模型使用方式以及适用范围

外呼场景中的对话辱骂风险识别。

### 如何使用

你可以使用StructBERT辱骂风险识别-中文-外呼-tiny模型，对外呼对话数据进行辱骂风险识别。 输入一段对话，模型会给出该对话的辱骂风险分类标签（{'无风险': 1, '辱骂风险': 0}）以及相应的概率。

#### 代码范例
<!--- 本session里的python代码段，将被ModelScope模型页面解析为快速开始范例--->
```python
from modelscope.pipelines import pipeline

input = '你好，阿里巴巴。'
classifier = pipeline('text-classification', model='damo/nlp_structbert_abuse-detect_chinese-tiny')
result = classifier(input)

print('输入文本:\n{}\n'.format(input))
print('分类结果:\n{}'.format(result))
```

### 模型局限性以及可能的偏差
如下训练数据介绍部分，训练数据集中存在数据类别分布不平衡的问题，即大量对话类别都是无辱骂风险，有辱骂风险的数据较少，该分类模型可能将一些有辱骂风险的对话错分为无风险。

## 训练数据介绍
每类数据的分布以及训练集与测试集中每类的具体分布如下：

```json
{
  "all_dataset": {
    "无风险": 41284,
    "辱骂风险": 722
  },
  "train_dataset": {
    "无风险": 37155,
    "辱骂风险": 649
  },
  "dev_dataset": {
    "无风险": 4129,
    "辱骂风险": 73
  }
}
```

## 数据评估及结果
micro f1=0.9938  macro f1=0.9118


### 相关论文以及引用信息
```
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```