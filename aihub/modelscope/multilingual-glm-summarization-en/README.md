
# 多语言英文摘要模型介绍

## 模型描述

本模型在预训练的多语言通用语言模型（Multilingual GLM 或 mGLM<sup>[1](#refer1)</sup>）基础之上微调而来，可支持对101种不同语言的长文本做英文摘要。多语言通用语言模型（mGLM）基于通用语言模型（GLM<sup>[2](#refer2)</sup>）框架，是一个在海量多语言文本语料上预训练而来的大规模语言模型，支持101种语言、25万种词例，模型参数量大小在1B左右。本模型基于SwissArmyTransformer<sup>[3](#refer3)</sup>和DeepSpeed<sup>[4](#refer4)</sup>开发。

## 期望模型使用方式以及适用范围 

本模型期望按照提示学习的方式来使用，通过添加特定的提示词引导模型输出目标的英文摘要。

**输入**：`{源文本} TL;DR: [sMASK] `

**输出**：`{源文本} TL;DR: {生成的摘要}`

**提示已经内嵌于平台代码内部，直接使用下文代码的范例即可直接调用**

### 代码范例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import MGLMSummarizationPreprocessor
from modelscope.utils.constant import Tasks

model = 'ZhipuAI/Multilingual-GLM-Summarization-en'
preprocessor = MGLMSummarizationPreprocessor()
pipe = pipeline(
    task=Tasks.text_summarization,
    model=model,
    preprocessor=preprocessor,
    model_revision='v1.0.1',
)
result = pipe(
    '据中国载人航天工程办公室消息，北京时间2022年10月25日，梦天实验舱与长征五号B遥四运载火箭组合体已转运至发射区。后续将按计划开展发射前各项功能检查和联合测试等工作，计划于近日择机实施发射。目前，文昌航天发射场设施设备状态良好，参试各单位正在加紧开展任务准备，全力以赴确保空间站建造任务决战决胜。'
)  
print(result)
```

### 模型局限性以及可能的偏差

1. 微调模型时所采用的语料覆盖面有限，虽然大规模预训练模型自身具有零样本迁移学习的能力，但其具体表现仍有一定的局限性，尤其是在语料欠缺的小语种表现上会有一定偏差，请用户自行评测后决定如何使用；
2. 微调数据集的领域分布主要集中在新闻领域和学术文章领域，由此也可能带来一定的偏差；
3. 模型基于自由生成的方式得到英文摘要，所以可能存在一定的几率输出不正确的事实；

## 训练数据介绍

本多语言英文摘要微调数据集主要由三部分构成：
1. **CNN/DailyMail**<sup>[5](#refer5)</sup>: 一个新闻摘要数据集，原始数据包含英文文章和英文摘要。为了让模型微调出跨语言的能力，我们将原文翻译成日语后使用；
2. **SciTLDR**<sup>[6](#refer6)</sup>: 一个英文学术论文摘要数据集，包含高质量的由作者或领域专家撰写的论文英文摘要；
3. **XWikis Corpus**<sup>[7](#refer7)</sup>: 一个基于维基百科的跨语言文本摘要数据集，支持四种语言：英语、德语、法语、捷克语；

## 模型训练流程

### 预处理

本模型的微调训练，在数据集的预处理过程包含了对原文最大长度截断和添加提示词与[sMASK]掩码等操作。

### 训练

本模型的微调训练使用**mGLM源码**<sup>[8](#refer8)</sup>中的`ds_finetune_summary.sh`和`seq_ensum.sh`等配置对mGLM模型进行微调训练。


### 相关论文以及引用信息

<div id="refer1"><a href="https://models.aminer.cn/mglm-1b/">[1] mGLM-1B: 开源的多语言预训练模型</a></div>
<div id="refer2"><a href="https://aclanthology.org/2022.acl-long.26.pdf">[2] GLM: General Language Model Pretraining with Autoregressive Blank Infilling</a></div>
<div id="refer3"><a href="https://github.com/THUDM/SwissArmyTransformer">[3] SwissArmyTransformer</a></div>
<div id="refer4"><a href="https://github.com/microsoft/DeepSpeed">[4] DeepSpeed Software Suite</a></div>
<div id="refer5"><a href="https://proceedings.neurips.cc/paper/2015/file/afdec7005cc9f14302cd0474fd0f3c96-Paper.pdf">[5] Teaching machines to read and comprehend</a></div>
<div id="refer6"><a href="https://arxiv.org/pdf/2004.15011">[6] TLDR: Extreme summarization of scientific documents</a></div>
<div id="refer7"><a href="https://datashare.ed.ac.uk/handle/10283/4188">[7] The XWikis Corpus (Perez-Beltrachini and Lapata, 2021)</a></div>
<div id="refer8"><a href="https://github.com/thudm/multilingual-glm">[8] Multilingual GLM Source Code</a></div>