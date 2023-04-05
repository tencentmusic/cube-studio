
# GPT3中文1.3B参数量文本生成模型
GPT-3模型是一个通用的预训练生成模型，使用Transformer的Decoder-only结构，可以用于解决下游各种类型的生成任务，特别是zero-shot生成能力。模型利用大量无监督数据，通过自回归任务进行预训练。可以用于解决文本生成相关的任务包含：文本摘要、问题生成、data-to-text等。

## 模型描述
GPT-3模型使用Transformer的Decoder结构，并对Transformer Decoder进行了一些改动，原本的Decoder包含了两个 Multi-Head Attention 结构，GPT-3只保留了 Mask Multi-Head Attention，利用常规的语言建模优化，从左到右的自回归预训练。本模型是基于GPT-3的代码结合大量中文无监督数据和下游任务数据预训练得到，我们训练了多种不同参数的模型
，GPT-3模型介绍，详见：[Language Models are Few-Shot Learners
](https://arxiv.org/abs/2005.14165)

本项目我们复现了一系列不同规模的中文GPT3模型，包括base/large/1.3B/2.7B/13B/30B/175B等，本模型是其中1.3B的版本。全部版本如下表所示：

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
本模型可直接用于文本生成，也可以通过finetune用于各类文本理解的任务。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope library之后即可使用GPT-3的text-generation的能力。目前我们免费提供试用的Notebook环境，使用的是单卡GPU，由于显存限制仅可以运行pipeline推理，如果用户在使用Notebook环境时想要运行训练，推荐使用更小规模的[large](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_chinese-large/summary)/[base](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_chinese-base/summary)版本

#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    input = '程序员脱发用什么洗发水'
    model_id = 'damo/nlp_gpt3_text-generation_1.3B'
    pipe = pipeline(Tasks.text_generation, model=model_id)

    print(pipe(input))
```

### 模型局限性以及可能的偏差
模型训练数据来源于网络，生成结果可能存在一定偏差。

## 训练数据介绍
训练数据包括中文维基百科、网络上公开文本数据。

## 模型训练流程
本模型的预训练分为两个阶段。第一阶段严格按照原始GPT3的参数设置进行：在中文wiki/ Common crawl等无监督数据上，通过自回归的训练任务训练了约300B字得到。第二阶段中，我们加入了多种有监督数据继续训练，使得模型具备多种任务的zero-shot的处理能力。

我们为GPT3模型支持了续写训练与输入输出形式的训练，训练方式不需要额外指定，训练数据集仅包含 src_txt 时会进行续写训练，同时包含 src_txt 和 tgt_txt 时会进行输入输出形式的训练。以下将为两种训练方式提供示例代码。

### 训练准备（重要）
目前对于GPT3 [1.3B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_1.3B/summary)/[2.7B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_2.7B/summary) 两个模型我们在训练阶段支持了运行时的模型自动拆分功能，因此不需要使用与训练tensor并行度匹配的checkpoint即可开始训练。需要注意的是，当使用并行训练时，务必提前在configuration.json中确认以下参数配置正确：

```json
"megatron": {
    "checkpoint_tensor_model_parallel_size": 1, # 对应checkpoint的并行片数，在1.3B/2.7B模型中为1
    "world_size": 1, # 全局的并行进程数
    "tensor_model_parallel_size": 1 # tensor 并行度
}
```

以单机8卡训练，2的数据并行度和4的tensor并行度（2dp+4tp）为例：

```shell
# 训练启动命令
torchrun --nproc_per_node 8 finetune_poetry.py # 这里的8是启动进程数，为dp*tp的值（2*4=8），单机训练时对应配置文件中的world_size
```

```json
# 无需配置数据并行度，会根据`world_size/tensor_model_parallel_size`计算
# 此时的正确配置
"megatron": {
    "checkpoint_tensor_model_parallel_size": 1, # 使用modelscope上传的checkpoint时无需修改
    "world_size": 8, # 此处对应上文的启动进程数nproc_per_node，如果使用其他方式启动多进程训练同理
    "tensor_model_parallel_size": 2 # 训练使用的 tensor 并行度
}
```

### 续写训练
下面是基于GPT-3中文1.3B模型在诗词生成数据集上二次开发训练

```python
# finetune_poetry.py
from torch.utils.tensorboard import SummaryWriter
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers


dataset_dict = MsDataset.load('chinese-poetry-collection')
train_dataset = dataset_dict['train'].remap_columns(
    {'text1': 'src_txt'})
eval_dataset = dataset_dict['test'].remap_columns({'text1': 'src_txt'})
max_epochs = 10
tmp_dir = './gpt3_poetry'

num_warmup_steps = 100

def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5),
               current_step * num_warmup_steps**(-1.5))

def cfg_modify_fn(cfg):
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': noam_lambda,
        'options': {
            'by_epoch': False
        }
    }
    cfg.train.optimizer = {'type': 'AdamW', 'lr': 3e-4}
    cfg.train.dataloader = {
        'batch_size_per_gpu': 16,
        'workers_per_gpu': 1
    }
    cfg.train.hooks.append({
        'type': 'MegatronHook'
    })
    cfg.evaluation.dataloader = {
        'batch_size_per_gpu': 8,
        'workers_per_gpu': 1
    }
    cfg.evaluation.metrics = 'ppl'
    return cfg

kwargs = dict(
    model='damo/nlp_gpt3_text-generation_1.3B',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_epochs=max_epochs,
    work_dir=tmp_dir,
    cfg_modify_fn=cfg_modify_fn)

# Construct trainer and train
trainer = build_trainer(
    name=Trainers.gpt3_trainer, default_args=kwargs)
trainer.train()
```

以上为单卡训练脚本，我们推荐使用 torchrun 拉起训练
```
torchrun finetune_poetry.py
```

### 输入输出形式训练
下面是基于GPT-3中文1.3B模型在Dureader问题生成数据集上二次开发训练

```python
# finetune_dureader.py
from torch.utils.tensorboard import SummaryWriter
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers


dataset_dict = MsDataset.load('DuReader_robust-QG')

train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}) \
    .map(lambda example: {'src_txt': example['src_txt'] + '\n'})
eval_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'}) \
    .map(lambda example: {'src_txt': example['src_txt'] + '\n'})

max_epochs = 10

tmp_dir = './gpt3_dureader'

num_warmup_steps = 200

def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5),
               current_step * num_warmup_steps**(-1.5))

def cfg_modify_fn(cfg):
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': noam_lambda,
        'options': {
            'by_epoch': False
        }
    }
    cfg.train.optimizer = {'type': 'AdamW', 'lr': 1e-4}
    cfg.train.dataloader = {
        'batch_size_per_gpu': 4,
        'workers_per_gpu': 1
    }
    cfg.train.hooks.append({
        'type': 'MegatronHook'
    })
    cfg.preprocessor.sequence_length = 512
    cfg.model.checkpoint_model_parallel_size = 1
    return cfg

kwargs = dict(
    model='damo/nlp_gpt3_text-generation_1.3B',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_epochs=max_epochs,
    work_dir=tmp_dir,
    cfg_modify_fn=cfg_modify_fn)

trainer = build_trainer(
    name=Trainers.gpt3_trainer, default_args=kwargs)
trainer.train()
```

以上为数据并行度为1的训练脚本，我们推荐使用 torchrun 拉起训练

单机单卡或单机多卡运行时，可以通过以下命令运行训练：
```
# N 为模型并行度
torchrun --nproc_per_node $N finetune_dureader.py
```

需要注意，目前1.3B参数量的GPT-3模型训练至少需要32G显存的gpu（如V100）才能进行单卡训练，或至少需要两张16G显存的gpu进行张量并行训练

### 推理加速
我们对大规模生成模型的推理速度进行了极致优化，13B模型128字的文本生成可以在1秒左右完成。

### 数据评估及结果

## 相关引用
我们将尽快推出本项目的技术报告，敬请期待！
