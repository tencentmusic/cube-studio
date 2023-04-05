
# UniTE介绍

## 模型描述

翻译质量评价，即对翻译文本进行质量评估，在给定源端输入、目标端参考答案、或两者均有提供的情况下，算法用于评估所生成文本的质量。本单一模型可同时支持提供源端输入（src-only）、目标端参考译文（ref-only）、或者两者均有（src-ref-combined）三种评价场景。
模型由一个预训练语言模型（Pretrained Language Model）和一个前馈神经网络（Feedforward Network）组成。模型首先在伪语料上进行继续预训练，而后在WMT'17-18 Metrics Shared Task数据集上进行微调。
此模型为large版本。

## 期望模型使用方式以及适用范围

包括提供源端输入（src-only）、目标端参考译文（ref-only）、或者两者均有（src-ref-combined）共三种文本质量评价场景。

### 如何使用

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.nlp.unite.configuration_unite import EvaluationMode

input = {
    'hyp': [
        'This is a sentence.',
        'This is another sentence.',
    ],
    'src': [
        '这是个句子。',
        '这是另一个句子。',
    ],
    'ref': [
        'It is a sentence.',
        'It is another sentence.',
    ]
}

pipeline_ins = pipeline(task=Tasks.translation_evaluation, model='damo/nlp_unite_up_translation_evaluation_English_large')
print(pipeline_ins(input))

# {'scores': [0.73634272813797, 0.7258641123771667]}

pipeline_ins.change_eval_mode(eval_mode=EvaluationMode.SRC)
print(pipeline_ins(input))

# {'scores': [-0.06169187277555466, 0.21349729597568512]}

pipeline_ins.change_eval_mode(eval_mode=EvaluationMode.REF)
print(pipeline_ins(input))

# {'scores': [0.8135091662406921, 0.8278040289878845]}

```

## 模型局限性以及可能的偏差
1. 模型在通用数据集上训练，部分垂直领域有可能产生一些偏差，请用户自行评测后决定如何使用。
2. 当前版本在单机单gpu环境测试通过，cpu和单机多gpu等其他环境待测试。

## 训练数据
WMT'17-18 Metrics Shared Task

## 训练流程
见论文

## 数据评估及结果
在WMT'19 Metrics Shared Task数据集上进行测试，计算与人工打分的Kendall's Tau系数。启动src-ref-combined评价功能，结果如下：

| *Method*        | Model Backbone | #Params. | En-Tgt | En-Src | En-Exc |  Avg   |
|:----------------|:--------------:|:--------:|:------:|:------:|:------:|:------:|
| BLEURT          |  BERT(En)      |  109M    |  33.1  |  -     |  -     |  -     |
| COMET           |  XLM-R-large   |  565M    |  34.5  |  56.6  |  42.2  |  45.6  |
| XLM-R+Concat    |  XLM-R-large   |  565M    |  33.5  |  56.7  |  44.1  |  45.6  |
| UniTE-MUP-base  |  XLM-R-base    |  283M    |  35.4  |  55.0  |  43.6  |  45.5  |
| UniTE-MUP-large |  XLM-R-large   |  565M    |**35.6**|**57.2**|**46.1**|**47.0**|

注：
- En-Tgt: English-targeted，即目标端输出为英语，共包括7个方向：De/Fi/Gu/Kk/Lt/Ru/Zh-En
- En-Src: English-sourced，即源端输入为英语，共包括8个方向：En-Cs/De/Fi/Gu/Kk/Lt/Ru/Zh
- En-Exc: English-excluded，即源端输入/目标端输出均不为英语，共包括3个方向：De-Cs, De-Fr, Fr-De
- All: 以上共18个方向的平均值

### 相关论文以及引用信息
``` bibtex
@inproceedings{wan-etal-2022-unite,
    title = "{U}ni{TE}: Unified Translation Evaluation",
    author = "Wan, Yu  and
      Liu, Dayiheng  and
      Yang, Baosong  and
      Zhang, Haibo  and
      Chen, Boxing  and
      Wong, Derek  and
      Chao, Lidia",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.558",
    doi = "10.18653/v1/2022.acl-long.558",
    pages = "8117--8127",,
}
```