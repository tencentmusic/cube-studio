
# PALM文本生成模型介绍
PALM模型是一个通用的预训练生成模型，可以用于解决下游各种类型的生成任务。模型利用大量无监督数据，通过结合自编码和自回归任务进行预训练。可以用于解决文本生成相关的任务包含：文本摘要、问题生成、data-to-text等。此处我们展示其中的一个应用，文本摘要生成：给定一个段落，生成相关核心摘要内容。本模型为PALM-摘要生成的Large模型，参数量约为5亿。


## 模型描述
针对实际场景中常见的文本生成需求，自主研发了PALM预训练语言生成模型。该模型通过在大规模文本上预训练得到，可作为下游自然语言生成任务的模型参数输入，以帮助提升下游任务的生成效果。PALM具有以下特点：

- 理解能力更强：为conditional generation特别设计了预训练任务，增强模型对上下文的理解能力。
- 所需标注更少：模型在海量文本语料上预训练，大大减少下游生成任务所需的标签数据量。
- 性能优良：中英文模型均使用大规模数据训练得到，且采用自研适应NLG任务的预训练目标。
- 适应各类生成任务：PALM可用于各类不同的生成任务，如摘要、问题生成、paraphrasing等等。
- 方便易用：下游使用方便，基于生成的传统encoder-decoder框架。

本模型是PALM通用预训练生成模型，在中文LCSTS数据集上进行finetune得到的文本摘要生成模型。PALM模型介绍，详见：[PALM:Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation](https://arxiv.org/abs/2004.07159)

![model](./resources/model.png)

### 相关模型

- [PALM 2.0预训练生成模型-中文-base](https://modelscope.cn/models/damo/nlp_palm2.0_pretrained_chinese-base/summary)：PALM2.0中文base通用生成模型，可以用于所有的中文生成场景的训练，如data-to-text，摘要生成等。
- [PALM 2.0摘要生成模型-中文-base](https://modelscope.cn/models/damo/nlp_palm2.0_text-generation_chinese-base/summary)：基于PALM2.0中文base模型训练得到的摘要生成模型
- [PALM 2.0商品文案生成-中文-base](https://modelscope.cn/models/damo/nlp_palm2.0_text-generation_commodity_chinese-base/summary)：基于PALM2.0中文base训练得到的商品文案描述生成模型
- [PALM 2.0天气生成模型-中文-base](https://modelscope.cn/models/damo/nlp_palm2.0_text-generation_weather_chinese-base/summary)：基于PALM2.0中文base训练得到的天气信息生成模型



## 期望模型使用方式以及适用范围
本模型主要用于给输入文档生成摘要内容。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成MaaS-lib之后即可使用text-generation的能力

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
text_generation_zh = pipeline(Tasks.text_generation, model='damo/nlp_palm2.0_text-generation_chinese-large')
result_zh = text_generation_zh("本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方："
"1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代")
print(result_zh['text'])
```

### 模型局限性以及可能的偏差
模型在数据集上训练，有可能产生一些偏差，请用户自行评测后决定如何使用。

## 训练数据介绍
本模型中文训练数据集是LCSTS，数据集240w左右， 具体数据可以[下载](http://icrc.hitsz.edu.cn/Article/show/139.html)

## 模型训练流程

### 训练
模型采用2张NVIDIA V100机器训练, 超参设置如下:
```
train_epochs=15
max_sequence_length=128
batch_size=4
learning_rate=1e-3
optimizer=AdamW
```

### 微调代码范例

```python
import tempfile

from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

# lcsts_test_set 为示例数据集，用户也可以使用自己的数据集进行训练
dataset_dict = MsDataset.load('lcsts_test_set', namespace='DAMO_NLP')

# 训练数据的输入出均为文本，需要将数据集预处理为输入为 src_txt，输出为 tgt_txt 的格式：
train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})
eval_dataset = dataset_dict['test'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})

# 用户自己数据集构造
# train_dataset_dict = {"src_txt": ["text1", "text2"], "tgt_txt": ["text1", "text2"]}
# eval_dataset_dict = {"src_txt": ["text1", "text2"], "tgt_txt": ["text1", "text2"]}
# train_dataset = MsDataset(Dataset.from_dict(train_dataset_dict))
# eval_dataset = MsDataset(Dataset.from_dict(eval_dataset_dict))

num_warmup_steps = 500
def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5),
               current_step * num_warmup_steps**(-1.5))

# 可以在代码修改 configuration 的配置
def cfg_modify_fn(cfg):
    cfg.preprocessor.sequence_length = 128
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
        "batch_size_per_gpu": 4,
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
模型在LCSTS测试数据评估结果

| Rouge-1  | Rouge-2 | Rouge-L |
|----------|-------- |---------|
| 46.58    | 31.98   | 42.61   |


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


