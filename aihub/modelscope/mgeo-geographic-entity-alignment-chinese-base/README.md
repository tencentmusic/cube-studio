
视频：[使用AI比较地址相似度](https://www.zhihu.com/zvideo/1608845874695696384)

教程：[使用Python+AI模型比较地址相似度](https://zhuanlan.zhihu.com/p/603106929)

## 快速传送
- [点我试用训练好的门址地址要素解析模型](https://modelscope.cn/models/damo/mgeo_geographic_elements_tagging_chinese_base/summary)
- [点我试用训练好的地理实体对齐模型](https://modelscope.cn/models/damo/mgeo_geographic_entity_alignment_chinese_base/summary)
- [点我试用训练好的Query-POI相关性排序](https://modelscope.cn/models/damo/mgeo_geographic_textual_similarity_rerank_chinese_base/summary)
- [点我试用训练好的地址Query成分分析模型](https://modelscope.cn/models/damo/mgeo_geographic_composition_analysis_chinese_base/summary)
- [点我试用训练好的WhereWhat切分模型](https://modelscope.cn/models/damo/mgeo_geographic_where_what_cut_chinese_base/summary)
- [点我查看海量地址处理应用案例源码](https://github.com/PhantomGrapes/MGeoExample)

## 版本要求
- modelscope版本大于等于1.2.0
- 推荐安装方式
```
# GPU版本
conda create -n py37testmaas python=3.7
pip install cryptography==3.4.8  tensorflow-gpu==1.15.5  torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
```
# CPU版本
conda create -n py37testmaas python=3.7
pip install cryptography==3.4.8  tensorflow==1.15.5  torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
- 免安装使用
```
点击右上角快速体验按钮，选在CPU或GPU机器运行实例，创建notebook执行推理或训练代码即可
```

第一次启动时可能会花费一些时间创建索引
- 如果您在使用中遇到任何困难，请钉钉搜索加入答疑群，群号：44837352（官方答疑）26735013283（技术交流）

## 任务介绍

日常生活中输入的地理文本可以为以下几种形式：

- 包含四级行政区划及路名路号POI的规范地理文本；
- 要素缺省的规范地理文本，例：只有路名+路号、只有POI；
- 非规范的地理文本、口语化的地理信息描述，例：阿里西溪园区东门旁亲橙里；


每一种地理文本的表述都可以映射到现实地理世界的道路、村庄、POI等实体上。地理实体对齐任务需要判断两段地理文本是否指代同一地理实体。该任务是构建地理信息知识库的核心技术。


多样化的地理文本写法对地理实体对齐任务提出的挑战如下：

- 同一个地理元素存在多种写法，没有给定的改写词表；
- 地理文本一般存在省市区等限制条件，需要结合限制条件分析相关性；
- 不同地市地理文本描述规范不一，对模型泛化性提出更高要求；


本任务需要输出两条地址的的对齐程度，分为完全对齐（exact_match）、部分对齐（partial_match）、不对齐（not_match）

## 模型描述
地址由于其丰富的表达以及与地图联动的多模态属性，一直是自动化处理的一个难题。达摩院联合高德发布了多任务多模态地址预训练底座MGeo模型。该模型基于地图-文本多模态架构，使用多任务预训练（MOMETAS）技术融合了注意力对抗预训练（ASA）、句子对预训练（MaSTS）、多模态预训练，训练得到适合于多类地址任务的预训练底座，为下游广泛的地址处理任务带来性能提升。
- MOMETAS：动态融合多种预训练目标得到普适性更强的预训练底座，技术发表于EMNLP2022（[论文链接](https://arxiv.org/abs/2210.10293)）。
- ASA：对预训练时的自注意力进行对抗攻击训练避免对局部信息的过多关注，技术发表于AAAI2023（[论文链接](https://arxiv.org/abs/2206.12608)）。
- MaSTS：更适合于捕捉句子对之间关系的预训练技术，登顶CLUE语义匹配榜首，通用版模型已开源（[模型链接](https://www.modelscope.cn/models/damo/nlp_masts_sentence-similarity_clue_chinese-large/summary)）。
- 地图-文本多模态预训练：业内首次实现对于地图的建模表示以及地图-文本的跨模态融合（[论文链接](https://arxiv.org/abs/2301.04283)）。

在ModelScope中我们开源的版本是基于开源地址数据以及开源地图OpenStreetMap训练的MGeo预训练底座以及其在GeoGLUE地理语义评测榜中多个任务的下游模型。

更多信息详见MGeo底座模型：https://modelscope.cn/models/damo/mgeo_backbone_chinese_base/summary

## 一键调用
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

task = Tasks.sentence_similarity
model = 'damo/mgeo_geographic_entity_alignment_chinese_base'
inputs = inputs = ('紫萱路363号人力社保局', '紫萱路363号市人社局')
# v1.2.0和v1.1.2均可尝试使用
pipeline_ins = pipeline(
    task=task, model=model, model_revision='v1.2.0')
print(pipeline_ins(input=inputs))
# 输出
# {'scores': [0.06451419740915298, 0.9217355251312256, 0.013750356622040272], 'labels': ['partial_match', 'exact_match', 'not_match']}
```

## 自定义训练

当用户有自己标注好的数据希望基于MGeo底座进行训练或基于训练好的下游模型进行继续训练时，可使用自定义训练功能。


以GeoGLUE的地理实体对齐任务为例
```python
import os
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers, Preprocessors
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.trainers import build_trainer

tmp_dir = 'tmp_dir'

def finetune(model_id,
             train_dataset,
             eval_dataset,
             name=Trainers.nlp_text_ranking_trainer,
             cfg_modify_fn=None,
             **kwargs):
    kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=tmp_dir,
        cfg_modify_fn=cfg_modify_fn,
        **kwargs)

    os.environ['LOCAL_RANK'] = '0'
    trainer = build_trainer(name=name, default_args=kwargs)
    trainer.train()
    results_files = os.listdir(tmp_dir)

def cfg_modify_fn(cfg):
    cfg.task = Tasks.sentence_similarity
    cfg['preprocessor'] = {'type': Preprocessors.sen_sim_tokenizer}

    cfg.train.dataloader.batch_size_per_gpu = 64
    cfg.evaluation.dataloader.batch_size_per_gpu = 64
    cfg.train.optimizer.lr = 2e-5
    cfg.train.max_epochs = 1

    cfg['dataset'] = {
        'train': {
            'labels': ['not_match', 'partial_match', 'exact_match'],
            'first_sequence': 'sentence1',
            'second_sequence': 'sentence2',
            'label': 'label',
            'sequence_length': 128
        }
    }
    cfg['evaluation']['metrics'] = "seq-cls-metric"
    cfg.train.hooks = [
    {
        'type': 'CheckpointHook',
        'interval': 1
    },
    {
        'type': 'TextLoggerHook',
        'interval': 100
    }, {
        'type': 'IterTimerHook'
    }, {
        'type': 'EvaluationHook',
        'by_epoch': True
    }]
    cfg.train.lr_scheduler.total_iters = int(len(train_dataset) / 32) * cfg.train.max_epochs
    return cfg

# load dataset
train_dataset = MsDataset.load('GeoGLUE', subset_name='GeoEAG', split='train', namespace='damo')
dev_dataset = MsDataset.load('GeoGLUE', subset_name='GeoEAG', split='validation', namespace='damo')

model_id = 'damo/mgeo_backbone_chinese_base'
finetune(
    model_id=model_id,
    train_dataset=train_dataset['train'],
    eval_dataset=dev_dataset['validation'],
    cfg_modify_fn=cfg_modify_fn,
    name='nlp-base-trainer')

output_dir = os.path.join(tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)
print(f'model is saved to {output_dir}')
```

如果需要从本地load用户自定义数据，可以先将数据处理为如下格式，并保存为train.json和dev.json：
```json
{"sentence1": "兰花小区四小区五幢五单元", "sentence2": "乌兰小区四区5栋乌兰小区4区5栋", "label": "not_match"}
```
如果只有一个句子，去掉sentence2字段即可。更新原始代码中的labels字段为新数据集的标签：
```
cfg['dataset'] = {
    'train': {
        'labels': ['not_match', 'partial_match', 'exact_match'],
        'first_sequence': 'sentence1',
        'second_sequence': 'sentence2',
        'label': 'label',
        'sequence_length': 128
    }
}
```

然后替换原流程中的train_dataset和dev_dataset为：
```python
local_train = 'train.json'
local_test = 'dev.json'
train_dataset = MsDataset.load('json', data_files={'train': [local_train]})
dev_dataset = MsDataset.load('json', data_files={'validation': [local_test]})
```

## 相关论文以及引用信息

```bib
@article{DBLP:journals/corr/abs-2210-10293,
  author    = {Hongqiu Wu and
               Ruixue Ding and
               Hai Zhao and
               Boli Chen and
               Pengjun Xie and
               Fei Huang and
               Min Zhang},
  title     = {Forging Multiple Training Objectives for Pre-trained Language Models
               via Meta-Learning},
  journal   = {CoRR},
  volume    = {abs/2210.10293},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2210.10293},
  doi       = {10.48550/arXiv.2210.10293},
  eprinttype = {arXiv},
  eprint    = {2210.10293},
  timestamp = {Mon, 24 Oct 2022 18:10:06 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2210-10293.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@misc{2301.04283,
Author = {Ruixue Ding and Boli Chen and Pengjun Xie and Fei Huang and Xin Li and Qiang Zhang and Yao Xu},
Title = {A Multi-Modal Geographic Pre-Training Method},
Year = {2023},
Eprint = {arXiv:2301.04283},
}
@misc{2206.12608,
Author = {Hongqiu Wu and Ruixue Ding and Hai Zhao and Pengjun Xie and Fei Huang and Min Zhang},
Title = {Adversarial Self-Attention for Language Understanding},
Year = {2022},
Eprint = {arXiv:2206.12608},
}
```
## 使用答疑
如果您在使用中遇到任何困难，请钉钉搜索加入答疑群，群号：26735013283