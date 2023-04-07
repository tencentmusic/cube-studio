
# PALM文本生成模型介绍
PALM预训练语言生成模型是针对实际场景中常见的文本生成需求所设计的一个模型。模型利用大量无监督数据，通过结合自编码和自回归任务进行预训练，更贴合下游生成任务所同时需要的理解和生成能力。

## 模型描述
针对实际场景中常见的文本生成需求，自主研发了PALM预训练语言生成模型。该模型通过在大规模文本上预训练得到，可作为下游自然语言生成任务的模型参数输入，以帮助提升下游任务的生成效果。PALM具有以下特点：

- 理解能力更强：为conditional generation特别设计了预训练任务，增强模型对上下文的理解能力。
- 所需标注更少：模型在海量文本语料上预训练，大大减少下游生成任务所需的标签数据量。
- 性能优良：中英文模型均使用大规模数据训练得到，且采用自研适应NLG任务的预训练目标。
- 适应各类生成任务：PALM可用于各类不同的生成任务，如摘要、问题生成、paraphrasing等等。
- 方便易用：下游使用方便，基于生成的传统encoder-decoder框架。

本模型是PALM通用预训练生成模型，在英文CNN/Dail Mail上进行finetune得到的英文文本摘要生成模型。PALM模型介绍，详见：[PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation](https://arxiv.org/abs/2004.07159)

![model](./resources/model.png)

## 期望模型使用方式以及适用范围
本模型主要用于给输入文档生成摘要内容。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope-library之后即可使用text-generation的能力

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

text_generation_en = pipeline(Tasks.text_generation, model='damo/nlp_palm2.0_text-generation_english-base')
result_en = text_generation_en("The Director of Public Prosecutions who let off Lord Janner over alleged child sex abuse started"
"her career at a legal chambers when the disgraced Labour peer was a top QC there . Alison Saunders ,"
"54 , sparked outrage last week when she decided the 86-year-old should not face astring of charges"
"of paedophilia against nine children because he has dementia . Today , newly-released documents"
"revealed damning evidence that abuse was covered up by police andsocial workers for more than 20 years ."
"And now it has emerged Mrs Saunders ' law career got off to a flying start when she secured her"
"pupillage -- a barrister 's training contract at 1 Garden Court Chambers in London in 1983 .")

print(result_en['text'])
```

### 模型局限性以及可能的偏差
模型在数据集上训练，有可能产生一些偏差，请用户自行评测后决定如何使用。

## 训练数据介绍
本模型英文训练数据集是CNN/Daily Mail，数据集28w左右， 具体数据可以[下载](https://huggingface.co/datasets/cnn_dailymail/viewer/3.0.0/train)

## 模型训练流程

### 训练
模型采用2张NVIDIA V100机器训练, 超参设置如下:
```
train_epochs=15
max_sequence_length=512
batch_size=32
learning_rate=1e-3
optimizer=Adam
```

### 微调代码范例
在ModelScope上暂未上传英文PALM预训练模型，目前不支持微调。

## 数据评估及结果
PALM在摘要数据集上的效果：

![palm](./resources/palm.png)


### 相关论文以及引用信息
如果我们的模型对您有帮助，请您引用我们的文章：
```
@inproceedings{bi-etal-2020-palm,
    title = "{PALM}: Pre-training an Autoencoding & Autoregressive Language Model for Context-conditioned Generation",
    author = "Bi, Bin  and
      Li, Chenliang  and
      Wu, Chen  and
      Yan, Ming  and
      Wang, Wei  and
      Huang, Songfang  and
      Huang, Fei  and
      Si, Luo",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.700",
    doi = "10.18653/v1/2020.emnlp-main.700",
    pages = "8681--8691"}
```
