
# 纠错模型介绍
文本纠错任务检测并纠正句子中存在的拼写、语法、语义等错误，在搜索，写作办公、教育等场景具有重要意义。文书校对评测任务应用纠错技术以辅助司法人员自动检出并纠正法律文书中存在的错误，涵盖了法律文书中存在的别字、冗余、缺失、乱序四种类型的错误。

基于seq2seq模型，我们通过数据增强方式缓解纠错训练数据稀缺问题，并在各个阶段取得了第一名(LegalMind-GEC队伍)。详见法研杯2022[文书校对赛道](http://cail.cipsc.org.cn/task2.html?raceID=2&cail_tag=2022#)
<p align="center">
    <img src="./description/final_rank_fyb.jpg" alt="donuts" />

## 模型描述

模型结构上，如下图所所示，我们采用基于transformer的seq2seq方法建模文本纠错任务，输入待纠错文本，输出正确文本。

训练数据上，主办方共提供了约1万条标注数据用于训练，对于seq2seq模型而言，数据规模是远远不够的。我们结合字、词混淆集，采用数据增强方法构造了约200万的伪训练语料。

模型训练上，我们根据训练数据的质量，进行了多阶段的训练。首先，我们使用中文BART初始化模型，然后在法律领域伪纠错数据上训练进行领域适应，最后我们分别在蜜度和官方语料人工标注语料上进行最终的训练。

效果上，我们最终多模型编辑集成系统，在第二阶段分数81.81，该模型（单模型）分数80.12。

<p align="center">
    <img src="./description/model.jpg" alt="donuts" />

模型效果如下：   
输入：2012年、2013年收入统计表复印件各一份，欲证明被告未足额**知府**社保费用。   
输出：2012年、2013年收入统计表复印件各一份，欲证明被告未足额**支付**社保费用。 

## 期望模型使用方式以及适用范围
本模型主要用于对中文文本进行错误诊断，输出符合拼写、语法要求的文本。
模型输入上，该纠错模型是一个句子级别的模型，模型效果会受到文本长度、分句粒度的影响，建议是每次输入一句话。
具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope library之后即可使用法律领域的text-error-correction的能力

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/nlp_bart_text-error-correction_chinese-law'
input = '2012年、2013年收入统计表复印件各一份，欲证明被告未足额知府社保费用。'
pipeline = pipeline(Tasks.text_error_correction, model=model_id, model_revision='v1.0.1')
result = pipeline(input)
print(result['output'])

```

### 模型局限性以及可能的偏差
模型在法律领域进行训练，不同场景下有可能产生一些偏差，请用户自行评测后决定如何使用。

## 训练数据介绍
本模型用到了大规模的伪纠错数据和高质量的人工标注数据，伪纠错数据披露后公开，人工标注数据可以评测[官网下载](http://cail.cipsc.org.cn/task2.html?raceID=2&cail_tag=2022#)

## 模型训练流程
暂不支持在ModelScope内部进行训练

## 数据评估及结果
我们给出法研杯2022文书校对赛道第二阶段前三名的成绩
|                   | detection-F1     | correction-F1     | score  |
|-------------------|-------|-------|-------|
| LegalMind-GEC (our)|  82.413| 79.397 | 81.81 |
| TAL-火眼金睛 | 81.48 | 75.236	 | 80.231 |
| 婷之队           | 77.028 | 67.281 | 75.079 |
| 本模型 (our single model) | 80.834 | 77.252 | 80.118 |
||||


### 相关论文以及引用信息
我们的模型训练和编辑级别投票集成代码已经开源在：https://github.com/HillZhang1999/MuCGEC 。 如使用相应方案，请引用我们的论文：
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
