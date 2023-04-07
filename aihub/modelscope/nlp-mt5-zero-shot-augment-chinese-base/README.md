
## 全任务零样本学习模型（mT5分类增强版）

该模型在mt5模型基础上使用了大量中文数据进行训练，并引入了零样本分类增强的技术，使模型输出稳定性大幅提升。

支持任务包含：
1. 文本分类：给定一段文本和候选标签，模型可输出文本所属的标签。
2. 自然语言推理：给定两段文本，判断两者关系。
3. 阅读理解：给定问题和参考文本，输出问题的答案。
4. 问题生成：给定答案和参考文本，生成该答案对应的问题。
5. 摘要生成：给定一段文本，生成该文本的摘要。
6. 标题生成：给定一段文本，为其生成标题。
7. 评价对象抽取：给定一段文本，抽取该段文本的评价对象。
8. 翻译：给定一段文本，将其翻译成另一种语言。

## 模型描述

该模型在mT5模型的基础上，使用3000万数据进行全中文任务的训练，支持各类任务的零样本/少样本学习。
模型特点：
1. 零样本分类增强：该针对零样本分类输出不稳定的情况（即生成的内容不在所给的标签之中），进行了数据增强，在零样本分类稳定性评测中，该模型输出稳定性可达98.51%。
2. 任务统一框架：把所有任务，如文本分类、相似度计算、文本生成等，都使用一个text-to-text的框架进行解决。

如有零样本分类、自动标注的需求也可以参考使用以下模型：

tiny版本模型：[StructBERT零样本分类-中文-tiny](https://www.modelscope.cn/models/damo/nlp_structbert_zero-shot-classification_chinese-tiny/summary)

base版本模型[**推荐**]：[StructBERT零样本分类-中文-base](https://www.modelscope.cn/models/damo/nlp_structbert_zero-shot-classification_chinese-base/summary)

large版本模型：[StructBERT零样本分类-中文-large](https://www.modelscope.cn/models/damo/nlp_structbert_zero-shot-classification_chinese-large/summary)


## 期望模型使用方式及适用范围

### 如何使用

在安装完成ModelScope之后即可使用该模型能力。

#### 代码范例
```python
from modelscope.pipelines import pipeline

t2t_generator = pipeline("text2text-generation", "damo/nlp_mt5_zero-shot-augment_chinese-base", model_revision="v1.0.0")

print(t2t_generator("文本分类。\n候选标签：故事,房产,娱乐,文化,游戏,国际,股票,科技,军事,教育。\n文本内容：他们的故事平静而闪光，一代人奠定沉默的基石，让中国走向繁荣。"))
# {'text': '文化'}

print(t2t_generator("抽取关键词：\n在分析无线Mesh网路由协议所面临挑战的基础上,结合无线Mesh网络的性能要求,以优化链路状态路由(OLSR)协议为原型,采用跨层设计理论,提出了一种基于链路状态良好程度的路由协议LR-OLSR.该协议引入了认知无线网络中的环境感知推理思想,通过时节点负载、链路投递率和链路可用性等信息进行感知,并以此为依据对链路质量进行推理,获得网络中源节点和目的节点对之间各路径状态良好程度的评价,将其作为路由选择的依据,实现对路由的优化选择,提高网络的吞吐量,达到负载均衡.通过与OLSR及其典型改进协议P-OLSR、SC-OLSR的对比仿真结果表明,LR-OLSB能够提高网络中分组的递交率,降低平均端到端时延,在一定程度上达到负载均衡."))
# {'text': '无线Mesh网,路由协议,环境感知推理'}

print(t2t_generator("为以下的文本生成标题：\n在分析无线Mesh网路由协议所面临挑战的基础上,结合无线Mesh网络的性能要求,以优化链路状态路由(OLSR)协议为原型,采用跨层设计理论,提出了一种基于链路状态良好程度的路由协议LR-OLSR.该协议引入了认知无线网络中的环境感知推理思想,通过时节点负载、链路投递率和链路可用性等信息进行感知,并以此为依据对链路质量进行推理,获得网络中源节点和目的节点对之间各路径状态良好程度的评价,将其作为路由选择的依据,实现对路由的优化选择,提高网络的吞吐量,达到负载均衡.通过与OLSR及其典型改进协议P-OLSR、SC-OLSR的对比仿真结果表明,LR-OLSB能够提高网络中分组的递交率,降低平均端到端时延,在一定程度上达到负载均衡."))
# {'text': '基于链路状态良好程度的无线Mesh网路由协议'}

print(t2t_generator("为下面的文章生成摘要：\n据统计，今年三季度大中华区共发生58宗IPO交易，融资总额为60亿美元，交易宗数和融资额分别占全球的35%和25%。报告显示，三季度融资额最高的三大证券交易所分别为东京证券交易所、深圳证券交易所和马来西亚证券交易所"))
# {'text': '大中华区IPO融资额超60亿美元'}

print(t2t_generator("评价对象抽取：颐和园还是挺不错的，作为皇家园林，有山有水，亭台楼阁，古色古香，见证着历史的变迁。"))
# {'text': '颐和园'}

print(t2t_generator("翻译成英文：如果日本沉没，中国会接收日本难民吗？"))
# {'text': 'will China accept Japanese refugees if Japan sinks?'}

print(t2t_generator("情感分析：外观漂亮，性能不错，屏幕很好。"))
# {'text': '积极'}

print(t2t_generator("根据给定的段落和答案生成对应的问题。\n段落：跑步后不能马上进食，运动与进食的时间要间隔30分钟以上。看你跑步的量有多大。不管怎么样，跑完步后要慢走一段时间，将呼吸心跳体温调整至正常状态才可进行正常饮食。血液在四肢还没有回流到内脏，不利于消化，加重肠胃的负担。如果口渴可以喝一点少量的水。洗澡的话看你运动量。如果跑步很剧烈，停下来以后，需要让身体恢复正常之后，再洗澡，能达到放松解乏的目的，建议15-20分钟后再洗澡；如果跑步不是很剧烈，只是慢跑，回来之后可以马上洗澡。 \n 答案：30分钟以上"))
# {'text': '跑步后多久进食'}
```

finetune代码范例
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
    model='damo/nlp_mt5_zero-shot-augment_chinese-base',
    model_revision="v1.0.2",
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=tempfile.TemporaryDirectory().name,
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(
    name=Trainers.text_generation_trainer, default_args=kwargs)
trainer.train()


```

### 模型局限性及可能的偏差
我们的模型基于大规模NLP数据集进行训练得到，在各领域中都表现出了良好性能，但在一些垂直领域可能表现稍弱，可对模型进一步finetune进行下游运用。

## 数据评估及结果

### 零样本分类稳定性评测
#### 评测原因
在进行零样本分类时，即按照一定格式拼接文本和候选标签进行分类任务，模型输出的结果可能并不在所给的候选标签之中，导致模型不可用。

#### 评测目的
通过稳定性评测，对模型零样本分类的稳定性进行量化，获取每个模型在随机标签、随机文本的情况下的稳定性指标。

#### 评测方式
从40万条文本中随机挑选了1万条文本作为待分类文本，再为每条文本从130个标签库中随机挑选随机数量的标签作为候选标签，最后结合文本和候选标签得到评测数据集。

对于每个模型均使用其在训练时使用的prompt构建模型输入。如果模型最终的输出存在于候选标签中，则认为该模型在该样本上的预测稳定，否则认为模型不稳定。

#### 评测结果

|                     **模型名字**                      | **零样本分类稳定率(%)** | 
|:-------------------------------------------------:|:---------------:| 
|                    PromptCLUE                     |      48.65      |
|               PromptCLUE-base-v1-5                |      76.32      |
|       nlp_mt5_zero-shot-augment_chinese-base      |      **98.51**      |

### pCLUE评测

|                     **模型名字**                   | **Score** | **阅读理解(F1)** | **阅读理解(EM)** | **分类(acc)** | **推理(acc)** | **生成(rouge-l)** | 
|:-------------------------------------------------:|:---------:|:------------:|:------------:|:-----------:|:-----------:|:---------------:| 
|                    PromptCLUE                     |   0.495   |    0.650     |    0.518     |    0.539    |    0.515    |      0.342      |
|       nlp_mt5_zero-shot-augment_chinese-base      | **0.528** |  **0.685**   |  **0.560**   |  **0.582**  |  **0.550**  |    **0.357**    |


## 相关论文以及引用信息
```
@article{2020t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {140},
  pages   = {1-67},
  url     = {http://jmlr.org/papers/v21/20-074.html}
}
```