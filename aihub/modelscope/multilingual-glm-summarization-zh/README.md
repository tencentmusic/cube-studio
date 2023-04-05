
# 多语言中文摘要模型介绍

## 模型描述

本模型在预训练的多语言通用语言模型（Multilingual GLM 或 mGLM<sup>[1](#refer1)</sup>）基础之上微调而来，可支持对101种不同语言的长文本做中文摘要。多语言通用语言模型（mGLM）基于通用语言模型（GLM<sup>[2](#refer2)</sup>）框架，是一个在海量多语言文本语料上预训练而来的大规模语言模型，支持101种语言、25万种词例，模型参数量大小在1B左右。本模型基于SwissArmyTransformer<sup>[3](#refer3)</sup>和DeepSpeed<sup>[4](#refer4)</sup>开发。

## 期望模型使用方式以及适用范围 

本模型期望按照提示学习的方式来使用，通过添加特定的提示词引导模型输出目标的中文摘要。

**输入**：`{源文本} 中文摘要：[sMASK] `

**输出**：`{源文本} 中文摘要：{生成的摘要}`

**提示已经内嵌于平台代码内部，直接使用下文代码的范例即可直接调用**

### 代码范例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import MGLMSummarizationPreprocessor
from modelscope.utils.constant import Tasks

model = 'ZhipuAI/Multilingual-GLM-Summarization-zh'
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

1. 微调模型时所采用的语料覆盖面有限，虽然大规模预训练模型自身具有零样本迁移学习的能力，但其具体表现仍有一定的局限性，尤其是在语料欠缺的小语种表现上。此外用于微调的中文摘要数据集主要来自于机器翻译，因此可能会有一定偏差，请用户自行评测后决定如何使用；
2. 微调数据集的领域分布主要集中在新闻领域和学术文章领域，由此也可能带来一定的偏差；
3. 模型基于自由生成的方式得到中文摘要，所以可能存在一定的几率输出不正确的事实；

## 训练数据介绍

本多语言中文摘要微调数据集主要由两部分构成：
1. **NCLS**<sup>[5](#refer5)</sup>: 一个跨语言摘要数据集，原数据集包含英文文章生成中文摘要和中文文章生成英文摘要两部分；
2. **SciTLDR**<sup>[6](#refer6)</sup>: 一个学术论文摘要数据集，包含高质量的由作者或领域专家撰写的论文摘要，我们将摘要翻译成中文后使用；

## 模型训练流程

### 预处理

本模型的微调训练，在数据集的预处理过程包含了对原文最大长度截断和添加提示词与[sMASK]掩码等操作。

### 训练

本模型的微调训练使用**mGLM源码**<sup>[7](#refer7)</sup>中的`ds_finetune_summary.sh`和`seq_ncls.sh`等配置对mGLM模型进行微调训练。

## 数据评估及结果

使用**Scisummnet**<sup>[8](#refer8)</sup>和**CNN/DailyMail** (CNNDM)<sup>[9](#refer9)</sup>摘要数据集对模型进行评估，我们将Scisummnet数据集的摘要翻译成中文；将CNNDM的原文翻译成多语言，将摘要翻译成中文。以下分别用Scisummnet-ZH和CNNDM-Multi2ZH表示。

|  模型   | 微调数据集  | Scisummnet-ZH | CNNDM-Multi2ZH |
|  ----  | ----  | ----  | ---- |
| mGLM-1B  | CNNDM | 32.4/8.5/18.6 | 29.6/12.6/19.9 |
| mGLM-1B  | NCLS | 35.7/10.6/20.2 | 28.2/10.9/19.5 |
| mGLM-1B  | NCLS + SciTLDR | 44.6/18.2/27.9 | 36.6/15.2/24.2 |

### 相关论文以及引用信息

<div id="refer1"><a href="https://models.aminer.cn/mglm-1b/">[1] mGLM-1B: 开源的多语言预训练模型</a></div>
<div id="refer2"><a href="https://aclanthology.org/2022.acl-long.26.pdf">[2] GLM: General Language Model Pretraining with Autoregressive Blank Infilling</a></div>
<div id="refer3"><a href="https://github.com/THUDM/SwissArmyTransformer">[3] SwissArmyTransformer</a></div>
<div id="refer4"><a href="https://github.com/microsoft/DeepSpeed">[4] DeepSpeed Software Suite</a></div>
<div id="refer5"><a href="https://arxiv.org/pdf/1909.00156">[5] NCLS: Neural cross-lingual summarization</a></div>
<div id="refer6"><a href="https://arxiv.org/pdf/2004.15011">[6] TLDR: Extreme summarization of scientific documents</a></div>
<div id="refer7"><a href="https://github.com/thudm/multilingual-glm">[7] Multilingual GLM Source Code</a></div>
<div id="refer8"><a href="https://ojs.aaai.org/index.php/AAAI/article/view/4727/4605">[8] Scisummnet: A large annotated corpus and content-impact models for scientific paper summarization with citation networks</a></div>
<div id="refer9"><a href="https://proceedings.neurips.cc/paper/2015/file/afdec7005cc9f14302cd0474fd0f3c96-Paper.pdf">[9] Teaching machines to read and comprehend</a></div>