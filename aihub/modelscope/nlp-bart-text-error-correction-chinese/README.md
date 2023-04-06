
# 中文文本纠错模型介绍
输入一句中文文本，文本纠错技术对句子中存在拼写、语法、语义等错误进行自动纠正，输出纠正后的文本。主流的方法为seq2seq和seq2edits，常用的中文纠错数据集包括NLPCC18和CGED等，我们最新的工作提供了高质量、多答案的测试集MuCGEC。
## 模型描述

如图所示，我们采用基于transformer的seq2seq方法建模文本纠错任务。模型训练上，我们使用中文BART作为预训练模型，然后在Lang8和HSK训练数据上进行finetune。不引入额外资源的情况下，本模型在NLPCC18测试集上达到了SOTA。

<p align="center">
    <img src="./description/model.jpg" alt="donuts" />

模型效果如下：   
输入：这洋的话，下一年的福气来到自己身上。   
输出：这**样**的话，下一年的福气**就会**来到自己身上。 

## 期望模型使用方式以及适用范围
本模型主要用于对中文文本进行错误诊断，输出符合拼写、语法要求的文本。该纠错模型是一个句子级别的模型，模型效果会受到文本长度、分句粒度的影响，建议是每次输入一句话。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope library之后即可使用text-error-correction的能力

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

#初始化纠错pipeline
model_id = 'damo/nlp_bart_text-error-correction_chinese'
pipeline = pipeline(Tasks.text_error_correction, model=model_id, model_revision='v1.0.1')

#单条调用
input = '这洋的话，下一年的福气来到自己身上。'
result = pipeline(input)
print(result['output'])


#批量调用
inputs = ['这洋的话，下一年的福气来到自己身上。', '在拥挤时间，为了让人们尊守交通规律，派至少两个警察或者交通管理者。', '随着中国经济突飞猛近，建造工业与日俱增']
batch_out = pipeline(inputs, batch_size=2)
for result in batch_out:
    print(result['output'])

```

### 模型局限性以及可能的偏差
模型在Lang8和HSK数据集上训练，不同场景下有可能产生一些偏差，请用户自行评测后决定如何使用。

## 训练数据介绍
本模型训练数据集是Lang8(1,092,285句)和HSK(95,320句)。 Lang8[下载](http://tcci.ccf.org.cn/conference/2018/taskdata.php), HSK由于版权问题，无法提供下载链接，可自行获取。

## 模型训练流程
暂不支持在ModelScope内部进行训练

## 数据评估及结果
本模型在NLPCC18测试集上，采用M2Scorer[NLPCC18官方评测工具](https://github.com/nusnlp/m2scorer)评估，同等规模和训练数据的模型中取得了SOTA。
|                   | P     | R     | F0.5  |
|-------------------|-------|-------|-------|
| Tang et al., 2021<sup>1</sup>| 47.41 | 23.72 | 39.51 |
| Sun et al., 2021<sup>2</sup>  | 45.33 | 27.61 | 40.17 |
| Ours<sup>3</sup>            | 48.89 | 32.80 | 44.53 |

参考工作：
1. 苏大：Tang et al. Chinese grammatical error correction enhanced by data augmentation from word and character levels. 2021.
2. 北大 & MSRA & CUHK：Sun et al. A Unified Strategy for Multilingual Grammatical Error Correction with Pre-trained Cross-Lingual Language Model. 2021.
3. Ours：Zhang et al. MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for
Chinese Grammatical Error Correction. 2022.

### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用我们的文章：
```BibTeX
@inproceedings{zhang-etal-2022-mucgec,
    title = "{M}u{CGEC}: a Multi-Reference Multi-Source Evaluation Dataset for {C}hinese Grammatical Error Correction",
    author = "Zhang, Yue  and
      Li, Zhenghua  and
      Bao, Zuyi  and
      Li, Jiacheng  and
      Zhang, Bo  and
      Li, Chen  and
      Huang, Fei  and
      Zhang, Min",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.227",
    pages = "3118--3130"
    }
```


