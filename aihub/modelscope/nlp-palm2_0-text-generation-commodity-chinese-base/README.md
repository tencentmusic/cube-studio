
# PALM商品文案描述生成介绍
PALM模型是一个通用的预训练生成模型，可以用于解决下游各种类型的生成任务。模型利用大量无监督数据，通过结合自编码和自回归任务进行预训练。可以用于解决文本生成相关的任务包含：文本摘要、问题生成、data-to-text等。此处我们展示data-to-text中的一个应用，商品文案描述生成：给定商品和一些卖点词，生成和卖点相关的商品文案描述。本模型为PALM-摘要生成的base模型，参数量约为2.3亿。


## 模型描述
针对实际场景中常见的文本生成需求，自主研发了PALM预训练语言生成模型。该模型通过在大规模文本上预训练得到，可作为下游自然语言生成任务的模型参数输入，以帮助提升下游任务的生成效果。PALM具有以下特点：

- 理解能力更强：为conditional generation特别设计了预训练任务，增强模型对上下文的理解能力。
- 所需标注更少：模型在海量文本语料上预训练，大大减少下游生成任务所需的标签数据量。
- 性能优良：中英文模型均使用大规模数据训练得到，且采用自研适应NLG任务的预训练目标。
- 适应各类生成任务：PALM可用于各类不同的生成任务，如摘要、问题生成、paraphrasing等等。
- 方便易用：下游使用方便，基于生成的传统encoder-decoder框架。

本模型是PALM通用预训练生成模型，在中文商品文案描述数据集上进行finetune得到的商品文案描述生成模型。PALM模型介绍，详见：[PALM:Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation](https://arxiv.org/abs/2004.07159)

![model](./resources/model.png)

### 相关模型

- [PALM 2.0预训练生成模型-中文-base](https://modelscope.cn/models/damo/nlp_palm2.0_pretrained_chinese-base/summary)：PALM2.0中文base通用生成模型，本模型就是基于这个backbone训练得到，可以用于所有的中文生成场景的训练，如data-to-text，摘要生成等。
- [PALM 2.0摘要生成模型-中文-base](https://modelscope.cn/models/damo/nlp_palm2.0_text-generation_chinese-base/summary)：基于PALM2.0中文base模型训练得到的摘要生成模型
- [PALM 2.0摘要生成模型-中文-large](https://modelscope.cn/models/damo/nlp_palm2.0_text-generation_chinese-large/summary)：基于PALM2.0中文large 训练得到的摘要生成模型
- [PALM 2.0天气生成模型-中文-base](https://modelscope.cn/models/damo/nlp_palm2.0_text-generation_weather_chinese-base/summary)：基于PALM2.0中文base训练得到的天气信息生成模型



## 期望模型使用方式以及适用范围
本模型主要用于给输入文档生成摘要内容。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成MaaS-lib之后即可使用text-generation的能力

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
text_generation_zh = pipeline(Tasks.text_generation, model='damo/nlp_palm2.0_text-generation_commodity_chinese-base')
result_zh = text_generation_zh("垃圾桶，双层，可拆卸，加高，加高双层，把手，垃圾桶，内附，万向轮")
print(result_zh['text'])
```

### 模型局限性以及可能的偏差
模型在家居家装商品数据集上训练，在其他品类商品上可能有一些效果一般，输入形式可以参考测试集上的数据。

## 训练数据介绍
本模型中文训练数据集是商品文案描述数据集，包含40w左右数据，数据格式参考[商品文案描述生成](https://modelscope.cn/datasets/lcl193798/product_description_generation/files)

## 模型训练流程
用户可以基于这个商品文案模型在自己的摘要数据上做continue train，如果不是文案生成任务，请前往通用PALM生成模型进行训练：[PALM 2.0预训练生成模型-中文-base
](https://modelscope.cn/models/damo/nlp_palm2.0_pretrained_chinese-base/summary)
### 训练
模型采用2张NVIDIA V100机器训练, 超参设置如下:
```
train_epochs=15
max_sequence_length=128
batch_size=8
learning_rate=1e-3
optimizer=AdamW
```

### 微调代码范例
```python
import tempfile

from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

# DuReader_robust-QG 为示例数据集，用户也可以使用自己的数据集进行训练
dataset_dict = MsDataset.load('DuReader_robust-QG')

# 训练数据的输入出均为文本，需要将数据集预处理为输入为 src_txt，输出为 tgt_txt 的格式：
train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})
eval_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})

num_warmup_steps = 500
def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5),
               current_step * num_warmup_steps**(-1.5))

# 可以在代码修改 configuration 的配置
def cfg_modify_fn(cfg):
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': noam_lambda,
        'options': {
            'by_epoch': False
        }
    }
    cfg.train.optimizer = {
        "type": "AdamW",
        "lr": 1e-3,
        "options": {}
    }
    cfg.train.max_epochs = 15
    cfg.train.dataloader = {
        "batch_size_per_gpu": 8,
        "workers_per_gpu": 1
    }
    return cfg

kwargs = dict(
    model='damo/nlp_palm2.0_pretrained_chinese-base',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=tempfile.TemporaryDirectory().name,
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(
    name=Trainers.text_generation_trainer, default_args=kwargs)
trainer.train()
```

### 训练tips

* 超参数调节主要是lr和epoch，可以在cfg_modify_fn里修改
* 生成长度短的数据集训练轮数可以小一些，在10～20epoch之间，生成长度长的数据集需要更多的轮数，如30～50epoch
* 生成所需要的数据集量比较大，如果任务难度简单，则1w～10w即可，生成难度难的任务需要更多数据


## 数据评估及结果
模型在家居测试集数据评估结果

| Rouge-1  | Rouge-2 | Rouge-L |
|----------|-------- |---------|
|   59.14  | 40.32   | 46.67   |


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
