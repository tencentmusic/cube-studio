
<!--- 以下model card模型说明部分，请使用中文提供（除了代码，bibtex等部分） --->
# QEMind翻译质量评估-多语言-通用领域介绍
用于机器翻译质量评估，可对翻译结果进行打分，该模型的ensemble系统获[WMT2021机器翻译大赛质量评估DA任务](https://www.statmt.org/wmt21/quality-estimation-task.html) 冠军

## 模型描述
该模型基于HuggingFace框架训练，以XLMRoberta作为预训练模型，添加Regression Head用有标注的QE数据进行迁移学xi，训练数据为[WMT2021质量评估任务DA数据集](https://github.com/sheffieldnlp/mlqe-pe/tree/master/data/direct-assessments)。

## 期望模型使用方式以及适用范围
该模型可用于双语句对的的质量打分，输入为原文及机翻译文，输出为翻译质量分数，分值介于0~1之间，分数越高代表翻译质量越好。


### 如何使用
在安装ModelScope完成后即可使用


#### 代码范例
<!--- 本session里的python代码段，将被ModelScope模型页面解析为快速开始范例--->
```python
from modelscope.pipelines import pipeline
p = pipeline('sentence-similarity', model='damo/nlp_translation_quality_estimation_multilingual', device='cpu')
print(p({"source_text": "Love is a losing game", "target_text": "宝贝，人和人一场游戏"}))
```

### 模型局限性以及可能的偏差
该模型训练数据主要包含以下语向：英语-德语、英语-中文、罗马尼亚语-英语、爱沙尼亚语-英语、尼泊尔语-英语、僧伽罗语-英语、俄语-英语，在其它语向上具备zero-shot能力，其中在英语-捷克语、英语-日语、普什图语-英语、高棉语-英语上已经过测试，其它语向未经测试，效果可能存在偏差。

## 训练数据介绍
[WMT2021质量评估任务DA数据集](https://github.com/sheffieldnlp/mlqe-pe/tree/master/data/direct-assessments)，训练数据由<原文，机翻译文，人工打分>的三元组组成，训练时将输入的原文和机翻译文进行拼接，并使用XLMRoberta的SentencePiece模型进行分词，输出分数进行了归一化。

## 模型训练流程
将多个语向的训练数据混合，在一张16G V100的GPU上进行训练，batch-size为8，训练3个epoch。

## 数据评估及结果
该模型的ensemble系统QEMind在WMT2021测试集上的评测结果如下（其它语向可以尝试zero-shot，但效果未经评测）：

| 语向                         | Pearson r | RMSE   | MAE    |
|------------------------------|-----------|--------|--------|
| English-German               | 0.5666    | 0.5787 | 0.4317 |
| English-Chinese              | 0.6025    | 0.5804 | 0.4499 |
| Romanian-English             | 0.9082    | 0.3925 | 0.3160 |
| Estonian-English             | 0.8117    | 0.4882 | 0.3931 |
| Nepalese-English             | 0.8667    | 0.5698 | 0.4260 |
| Sinhala-English              | 0.5956    | 0.7825 | 0.6086 |
| Russian-English              | 0.8060    | 0.5338 | 0.3881 |
| English-Czech (zero-shot)    | 0.7462    | 0.5986 | 0.5986 |
| English-Japanese (zero-shot) | 0.3589    | 0.7574 | 0.5595 |
| Pashto-English (zero-shot)   | 0.6474    | 0.7364 | 0.6048 |
| Khmer-English (zero-shot)    | 0.6787    | 0.7293 | 0.5635 |

### 相关论文以及引用信息
如果您在您的研究中使用了该模型，请引用下面两篇论文

```bibtex
@inproceedings{wang-etal-2021-beyond-glass,
    title = "Beyond Glass-Box Features: Uncertainty Quantification Enhanced Quality Estimation for Neural Machine Translation",
    author = "Wang, Ke  and
      Shi, Yangbin  and
      Wang, Jiayi  and
      Zhang, Yuqi  and
      Zhao, Yu  and
      Zheng, Xiaolin",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.401",
    doi = "10.18653/v1/2021.findings-emnlp.401",
    pages = "4687--4698",
}
```

```bibtex
@inproceedings{wang-etal-2021-qemind,
    title = "{QEM}ind: {A}libaba{'}s Submission to the {WMT}21 Quality Estimation Shared Task",
    author = "Wang, Jiayi  and
      Wang, Ke  and
      Chen, Boxing  and
      Zhao, Yu  and
      Luo, Weihua  and
      Zhang, Yuqi",
    booktitle = "Proceedings of the Sixth Conference on Machine Translation",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wmt-1.100",
    pages = "948--954",
    abstract = "Quality Estimation, as a crucial step of quality control for machine translation, has been explored for years. The goal is to to investigate automatic methods for estimating the quality of machine translation results without reference translations. In this year{'}s WMT QE shared task, we utilize the large-scale XLM-Roberta pre-trained model and additionally propose several useful features to evaluate the uncertainty of the translations to build our QE system, named \textit{ \textbf{QEMind} }. The system has been applied to the sentence-level scoring task of Direct Assessment and the binary score prediction task of Critical Error Detection. In this paper, we present our submissions to the WMT 2021 QE shared task and an extensive set of experimental results have shown us that our multilingual systems outperform the best system in the Direct Assessment QE task of WMT 2020.",
}
```

