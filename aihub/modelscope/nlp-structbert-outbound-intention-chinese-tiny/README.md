

<!--- 以下model card模型说明部分，请使用中文提供（除了代码，bibtex等部分） --->

# StructBERT意图识别-中文-外呼-tiny
本模型基于StructBERT-tiny模型，使用外呼场景下的对话意图识别数据进行微调得到的。

## 模型描述

本模型是在中文预训练模型StructBERT的基础上使用外呼意图识别数据进行微调得到的。关于StructBERT的详细介绍可参见https://www.modelscope.cn/models/damo/nlp_structbert_backbone_base_std/summary 。

## 期望模型使用方式以及适用范围

外呼场景中的意图识别。

### 如何使用

你可以使用StructBERT意图识别-中文-外呼-tiny模型，对外呼对话数据进行意图识别。 输入一段对话，模型会给出该对话的意图分类标签（{'企业生产': 1, '催收催缴': 2, '营销': 0, '通知': 4, '验证码': 3}）以及相应的概率。

#### 代码范例
<!--- 本session里的python代码段，将被ModelScope模型页面解析为快速开始范例--->
```python
from modelscope.pipelines import pipeline

input = '天猫双十一大促，你想要的这里都有。'
classifier = pipeline('text-classification', model='damo/nlp_structbert_outbound-intention_chinese-tiny')
result = classifier(input)

print('输入文本:\n{}\n'.format(input))
print('分类结果:\n{}'.format(result))
```

### 模型局限性以及可能的偏差
如下训练数据介绍部分，训练数据集中存在数据类别分布不平衡的问题，该分类模型对一些出现频次较少的意图（如验证码）的识别效果可能会差一些。

## 训练数据介绍
每类数据的分布以及训练集与测试集中每类的具体分布如下：

```json
{
  "all_dataset": {
    "营销": 46061,
    "企业生产": 17826,
    "通知": 11406,
    "催收催缴": 2719,
    "验证码": 699
  },
  "train_dataset": {
    "营销": 41454,
    "企业生产": 16043,
    "通知": 10265,
    "催收催缴": 2447,
    "验证码": 629
  },
  "dev_dataset": {
    "营销": 4607,
    "企业生产": 1783,
    "通知": 1141,
    "催收催缴": 272,
    "验证码": 70
  }
}
```

## 数据评估及结果
micro f1=0.9525  macro f1=0.9121


### 相关论文以及引用信息
```
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
```