
# GPT-3文本生成模型介绍
GPT-3模型是一个通用的预训练生成模型，使用Transformer的Decoder-only结构，可以用于解决下游各种类型的生成任务，特别是zero-shot生成能力。模型利用大量无监督数据，通过自回归任务进行预训练。可以用于解决文本生成相关的任务包含：文本摘要、问题生成、data-to-text等。


## 模型描述
GPT-3模型使用Transformer的 Decoder结构，并对Transformer Decoder进行了一些改动，原本的Decoder包含了两个 Multi-Head Attention 结构，GPT-3只保留了 Mask Multi-Head Attention，利用常规的语言建模优化，从左到右的自回归预训练。本模型是基于GPT-3的代码结合大量中文无监督数据和下游任务数据预训练得到，我们训练了多种不同参数的模型，此处展示的是GPT-3 Large模型。GPT-3模型介绍，详见：[Language Models are Few-Shot Learners
](https://arxiv.org/abs/2005.14165)

本项目我们复现了一系列不同规模的中文GPT3模型，包括base/large/1.3B/2.7B/13B/30B/175B等，本模型是其中large的版本。全部版本如下表所示：

|Model|Layers|Heads|d_model|LR|Batch|
|---|---|---|---|---|---|
|[base](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_chinese-base/summary)|12|12|768|6.0e-4|0.5M|
|[large](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_chinese-large/summary)|24|16|1024|3.0e-4|0.5M|
|[1.3B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_1.3B/summary)|24|32|2048|2.0e-4|2M|
|[2.7B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_2.7B/summary)|32|32|2560|1.6e-4|2M|
|[13B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_13B/summary)|40|40|5120|1.0e-4|6M|
|[30B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_30B/summary)|48|56|7168|1.0e-4|6M|
|175B(work in process)|96|96|12288|1.2e-4|6M|


## 期望模型使用方式以及适用范围
本模型主要用于多种场景输入的生成和续写。比如用户可以自行尝试输入各种内容，然后让模型去回答、续写或者根据指令回复

### 如何使用
在安装完成ModelScope library之后即可使用GPT-3的text-generation的能力

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
text_generation_zh = pipeline(Tasks.text_generation, model='damo/nlp_gpt3_text-generation_chinese-large')
result_zh = text_generation_zh("随着计算机视觉的飞速发展,人脸识别技术已从简单场景发展到复杂场景,也即姿态、光照、表情、噪声、遮挡、化妆、年龄、种族、性别等差异化所呈现的复杂场景。尽管已有的人脸识别系统在特定约束环境下的识别成功率较高,")
print(result_zh['text'])
```

### 模型局限性以及可能的偏差
模型在数据集上训练，有可能产生一些偏差，请用户自行评测后决定如何使用。

## 训练数据介绍
训练数据包括中文维基百科、网络上公开文本数据。

## 模型训练流程

### 预处理

训练数据只需包含src_txt字段，推荐使用MsDataset包装后使用ModelScope的Trainer进行训练。
```python
import tempfile
from datasets import Dataset
from modelscope.msdatasets import MsDataset

# 模拟训练数据集
src_dataset_dict = {
    'src_txt': [
        '测试文本1', '测试文本2', '测试文本3'
    ]
}
src_dataset = MsDataset(Dataset.from_dict(src_dataset_dict))
max_epochs = 3
tmp_dir = tempfile.TemporaryDirectory().name
```

### 训练
下面是基于GPT-3中文large模型在诗词生成数据集上二次开发训练

```python
# 基于modelscope中文gpt3底座二次开发得到诗词生成模型代码

from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config
from modelscope.metainfo import Metrics, Trainers
from datasets import Dataset
from modelscope.msdatasets import MsDataset

dataset_dict = MsDataset.load('chinese-poetry-collection')
train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt'})
eval_dataset = dataset_dict['test'].remap_columns({'text1': 'src_txt'})
print (eval_dataset)
max_epochs = 10
tmp_dir = "./gpt3_poetry"

num_warmup_steps = 100
def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step ** (-0.5), current_step * num_warmup_steps ** (-1.5))

def cfg_modify_fn(cfg):
    cfg.train.lr_scheduler = {
        "type": "LambdaLR",
        "lr_lambda": noam_lambda,
        "options": {"by_epoch": False}
    }
    cfg.train.optimizer = {
        "type": "AdamW",
        "lr": 3e-4
    }
    cfg.train.dataloader = {"batch_size_per_gpu": 16, "workers_per_gpu": 1}
    return cfg

kwargs = dict(
    model='damo/nlp_gpt3_text-generation_chinese-large',
    train_dataset=train_dataset,
    eval_datase=eval_dataset,
    max_epochs=max_epochs,
    work_dir=tmp_dir,
    cfg_modify_fn=cfg_modify_fn)

# 构造 trainer 并进行训练
trainer = build_trainer(
    name=Trainers.nlp_base_trainer, default_args=kwargs)
trainer.train()
```

### 训练tips
- 训练lr设置可参考上述表格里面不同模型的设置
- 对于训练数据较长的场景，可适当增加训练epoch


### 相关论文以及引用信息
如果GPT-3模型对您有帮助，请您引用该模型的相关文章：
```
@inproceedings{NEURIPS2020_1457c0d6,
 author = {Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and Agarwal, Sandhini and Herbert-Voss, Ariel and Krueger, Gretchen and Henighan, Tom and Child, Rewon and Ramesh, Aditya and Ziegler, Daniel and Wu, Jeffrey and Winter, Clemens and Hesse, Chris and Chen, Mark and Sigler, Eric and Litwin, Mateusz and Gray, Scott and Chess, Benjamin and Clark, Jack and Berner, Christopher and McCandlish, Sam and Radford, Alec and Sutskever, Ilya and Amodei, Dario},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
 pages = {1877--1901},
 publisher = {Curran Associates, Inc.},
 title = {Language Models are Few-Shot Learners},
 url = {https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
