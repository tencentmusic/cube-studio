
# 春联生成模型
春联生成模型是达摩院AliceMind团队利用基础生成大模型在春联场景的应用，该模型可以通过输入两字随机祝福词，生成和祝福词相关的春联。


## AliceMind基础生成大模型介绍
AliceMind基础生成大模型包含中文GPT-3，PALM和PLUG等，模型通过在大规模文本上无监督预训练得到，相关模型已经应用于AIGC的多个场景。

### 中文GPT-3
GPT-3模型使用Transformer的 Decoder结构，从左到右的自回归预训练。我们基于GPT-3的代码结合大量中文无监督数据和下游任务数据预训练得到，同时训练了多种不同参数的模型，此处展示的是GPT-3 Large模型。GPT-3模型介绍，详见：[Language Models are Few-Shot Learners
](https://arxiv.org/abs/2005.14165)

#### 相关模型

- [GPT-3 large](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_chinese-large/summary)：GPT-3 中文large通用生成模型，可以用于生成场景的二次开发训练， zero-shot生成效果不如更大参数模型
- [GPT-3 2.7B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_2.7B/summary)：GPT-3 中文2.7B通用生成模型，具备一定的zero-shot生成能力
- [GPT-3 13B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_13B/summary)：GPT-3 中文13B通用生成模型，zero-shot生成能力覆盖范围更广
- [GPT-3 30B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_30B/summary)：GPT-3 中文30B通用生成模型，目前社区开放的最大模型，zero-shot生成效果



### PALM
针对实际场景中常见的文本生成需求，自主研发了PALM预训练语言生成模型。该模型通过在大规模文本上预训练得到，可作为下游自然语言生成任务的模型参数输入，以帮助提升下游任务的生成效果。PALM模型介绍，详见：[PALM:Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation](https://arxiv.org/abs/2004.07159)

#### 相关模型

- [PALM 2.0预训练生成模型-中文-base](https://modelscope.cn/models/damo/nlp_palm2.0_pretrained_chinese-base/summary)：PALM2.0中文base通用生成模型，本模型就是基于这个backbone训练得到，可以用于所有的中文生成场景的训练，如data-to-text，摘要生成等。
- [PALM 2.0摘要生成模型-中文-large](https://modelscope.cn/models/damo/nlp_palm2.0_text-generation_chinese-large/summary)：基于PALM2.0中文large 训练得到的摘要生成模型
- [PALM 2.0商品文案生成-中文-base](https://modelscope.cn/models/damo/nlp_palm2.0_text-generation_commodity_chinese-base/summary)：基于PALM2.0中文base训练得到的商品文案描述生成模型
- [PALM 2.0天气生成模型-中文-base](https://modelscope.cn/models/damo/nlp_palm2.0_text-generation_weather_chinese-base/summary)：基于PALM2.0中文base训练得到的天气信息生成模型

### PLUG
PLUG是有海量高质量中文文本预训练得到的理解和生成联合模型。PLUG的训练由两阶段组成。首先我们训练了一个24层的基于[StructBERT](https://arxiv.org/abs/1908.04577)的encoder，然后我们基于此训练了一个24+6层的[PALM](https://arxiv.org/pdf/2004.07159.pdf?fbclid=IwAR0BNl1IzR5bhcuEbyfNw2UN7MApHFoFP3BN40FKkW8x3bqolK_HilU293I) encoder-decoder。这使得模型既可以通过finetune来处理文本分类、序列标注等自然语言理解（NLU）任务，也可以用来处理自然语言生成（NLG）的任务。

#### 相关模型

- [PLUG预训练生成模型-中文-27B](https://modelscope.cn/models/damo/nlp_plug_text-generation_27B/summary)：PLUG中文27B通用生成模型，可以用于中文zero-shot生成场景，如小说续写，专业文稿撰写等。
- [PLUG通用问题生成模型-中文-27B](https://modelscope.cn/models/damo/nlp_plug_question-generation_27B/summary)：基于PLUG中文27B在问题生成数据集上训练，可以用于问题生成场景或者FAQ挖掘


## 期望模型使用方式以及适用范围
本模型主要用于给输入两字的愿望词，然后模型会生成春联，目前春联主要是和兔年相关，所以兔元素会相对丰富一些。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope-library之后即可使用text-generation的能力

#### 推理代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

input = '以关键词“五福”写一副春联'
p = pipeline(Tasks.text_generation, model='damo/spring_couplet_generation')
result = p(input)

print('输入文本:' + input + '\n')
print('生成结果' + result[OutputKeys.TEXT])
```

### 模型局限性以及可能的偏差
模型在愿望词相关的生成上效果会好一些，在非愿望词上会有一些效果上的偏差。

## 训练数据介绍
本模型中文训练数据集是收集的春联数据，在10w左右。

## 模型训练流程
用户可以基于这个春联模型在自己的春联数据上做continue train，如果不是春联任务，请前往通用PALM生成模型进行训练：[PALM 2.0预训练生成模型-中文-base
](https://modelscope.cn/models/damo/nlp_palm2.0_pretrained_chinese-base/summary)
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
        "batch_size_per_gpu": 8,
        "workers_per_gpu": 1
    }
    return cfg

kwargs = dict(
    model='damo/spring_couplet_generation',
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
