
# 基于序列建模的文本分割模型

该模型基于wiki-en公开语料训练，对未分割的长文本进行段落分割。提升未分割文本的可读性以
及下游NLP任务的性能。

## 模型描述

随着在线教学、会议等技术的扩展，口语文档的数量以会议记录、讲座、采访等形式不断增加。然而，经过自动语音识别（ASR）系统生成的长篇章口语文字记录缺乏段落等结构化信息，会显著降低文本的可读性，十分影响用户的阅读和信息获取效率。 此外，缺乏结构化分割信息对于语音转写稿下游自然语言处理（NLP）任务的性能也有较大的性能影响。

文档分割被定义为自动预测文档的段（段落或章节）边界。近年来，诸多研究者提出了许多基于神经网络的文本分割算法。比如， 当前文本分割的 state of the art (SOTA) 是 Lukasik等提出的基于BERT的cross-segment模型，将文本分割定义为逐句的文本分类任务。

然而，文档分割是一个强依赖长文本篇章信息的任务，逐句分类模型并不能很好的利用长文本的语义信息，导致模型性能有着明显的瓶颈。而层次模型面临计算量大，推理速度慢等问题。我们工作的目标是探索如何有效利用足够的上下文信息以进行准确分割以及在高效推理效率之间找到良好的平衡。

![model_structure](./modelStructure.jpg)


## 使用方式
- 直接输入长篇未分割文章，得到输出结果

## 模型局限性以及可能的偏差
- 模型采用公开语料进行训练，在某些特定领域文本上的分割性能可能会有影响。

## 训练数据
- 使用公开的中英文wiki数据: https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/

## 模型效果评估
- 选择positive precision、positive recall、 positive F1作为客观评价指标。
- 更多信息见参考论文。

![en_exp_res](./en_exp_res.jpg)

## 代码范例
```python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(
    task=Tasks.document_segmentation,
    model='damo/nlp_bert_document-segmentation_english-base')

result = p(documents='The Saint Alexander Nevsky Church was established in 1936 by Archbishop Vitaly (Maximenko) () on a tract of land donated by Yulia Martinovna Plavskaya.The initial chapel, dedicated to the memory of the great prince St. Alexander Nevsky (1220–1263), was blessed in May, 1936.The church building was subsequently expanded three times.In 1987, ground was cleared for the construction of the new church and on September 12, 1989, on the Feast Day of St. Alexander Nevsky, the cornerstone was laid and the relics of St. Herman of Alaska placed in the foundation.The imposing edifice, completed in 1997, is the work of Nikolaus Karsanov, architect and Protopresbyter Valery Lukianov, engineer.Funds were raised through donations.The Great blessing of the cathedral took place on October 18, 1997 with seven bishops, headed by Metropolitan Vitaly Ustinov, and 36 priests and deacons officiating, some 800 faithful attended the festivity.The old church was rededicated to Our Lady of Tikhvin.Metropolitan Hilarion (Kapral) announced, that cathedral will officially become the episcopal See of the Ruling Bishop of the Eastern American Diocese and the administrative center of the Diocese on September 12, 2014.At present the parish serves the spiritual needs of 300 members.The parochial school instructs over 90 boys and girls in religion, Russian language and history.The school meets every Saturday.The choir is directed by Andrew Burbelo.The sisterhood attends to the needs of the church and a church council acts in the administration of the community.The cathedral is decorated by frescoes in the Byzantine style.The iconography project was fulfilled by Father Andrew Erastov and his students from 1995 until 2001.')

print(result[OutputKeys.TEXT])
```

## 相关论文以及引用信息

```bib
@inproceedings{DBLP:conf/asru/ZhangCLLW21,
  author    = {Qinglin Zhang and
               Qian Chen and
               Yali Li and
               Jiaqing Liu and
               Wen Wang},
  title     = {Sequence Model with Self-Adaptive Sliding Window for Efficient Spoken
               Document Segmentation},
  booktitle = {{IEEE} Automatic Speech Recognition and Understanding Workshop, {ASRU}
               2021, Cartagena, Colombia, December 13-17, 2021},
  pages     = {411--418},
  publisher = {{IEEE}},
  year      = {2021},
  url       = {https://doi.org/10.1109/ASRU51503.2021.9688078},
  doi       = {10.1109/ASRU51503.2021.9688078},
  timestamp = {Wed, 09 Feb 2022 09:03:04 +0100},
  biburl    = {https://dblp.org/rec/conf/asru/ZhangCLLW21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
