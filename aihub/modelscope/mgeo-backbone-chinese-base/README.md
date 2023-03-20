
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

## 地址信息处理是什么

地址信息处理是对地址相关文本的自动化挖掘、理解与关联。这项技术广泛地应用在社会生活的各个场景之中。例如我们常用的地图软件中就用到了大量的地址信息处理技术来构建POI库，实现POI搜索与推荐；在外卖物流行业中，对于地址的解析、定位准确率的提升则直接带来运力成本的大量降低；目前在诸多挪车、外呼、报警等场景下也用上了地址自动化处理技术，大大节省了接线员定位事故发生地的时间；在零售行业中，地址也是会员体系管理的核心要素。
![应用场景](fig/task_intro.png)

## MGeo多任务多模态地址预训练底座
地址由于其丰富的表达以及与地图联动的多模态属性，一直是自动化处理的一个难题。达摩院联合高德发布了多任务多模态地址预训练底座MGeo模型。该模型基于地图-文本多模态架构，使用多任务预训练（MOMETAS）技术融合了注意力对抗预训练（ASA）、句子对预训练（MaSTS）、多模态预训练，训练得到适合于多类地址任务的预训练底座，为下游广泛的地址处理任务带来性能提升。
- MOMETAS：动态融合多种预训练目标得到普适性更强的预训练底座，技术发表于EMNLP2022（[论文链接](https://arxiv.org/abs/2210.10293)）。
- ASA：对预训练时的自注意力进行对抗攻击训练避免对局部信息的过多关注，技术发表于AAAI2023（[论文链接](https://arxiv.org/abs/2206.12608)）。
- MaSTS：更适合于捕捉句子对之间关系的预训练技术，登顶CLUE语义匹配榜首，通用版模型已开源（[模型链接](https://www.modelscope.cn/models/damo/nlp_masts_sentence-similarity_clue_chinese-large/summary)）。
- 地图-文本多模态预训练：业内首次实现对于地图的建模表示以及地图-文本的跨模态融合（[论文链接](https://arxiv.org/abs/2301.04283)）。

在ModelScope中我们开源的版本是基于开源地址数据以及开源地图OpenStreetMap训练的MGeo预训练底座以及其在GeoGLUE地理语义评测榜中多个任务的下游模型。

## MGeo模型结构
地址信息处理涵盖了多种的NLP任务，从输入的形式来看可以分为：
- 单句任务：输入是一条地址；
- 句子对任务：输入是两条地址；
- 多模态任务：输入是地址以及地图；

我们针对这三种输入形态设计了三种预训练任务：注意力对抗预训练（ASA）、句子对预训练（MaSTS）、多模态预训练。并用多任务预训练（MOMETAS）技术将这三种预训练任务进行动态组合，训练得到适合于多类地址任务的预训练底座。

![模型结构](fig/model_arch.png)

## 模型效果
我们使用网络公开的地理语义数据与开源地图训练了MGeo社区版，在GeoGLUE上评测了MGeo模型与当下一些主流预训练模型的效果，模型规模均为base。

|            | 门址地址要素解析 | 地理实体对齐 | Query-POI库召回 | Query-POI相关性排序 | 地址Query成分分析 | WhereWhat切分 |
|------------|------------------|--------------|-----------------|---------------------|-------------------|---------------|
| Bert       | 90.41            | 78.88        | 24.59           | 81.52               | 65.06             | 69.65         |
| Roberta    | 90.79            | 78.84        | 39.37           | 83.20               | 64.65             | 66.44         |
| Ernie      | 90.63            | 79.44        | 24.03           | 81.82               | 66.48             | 67.73         |
| Nezha      | 91.17            | 79.77        | 35.31           | 81.38               | 67.70             | 70.13         |
| StructBert | 91.22            | 78.83        | 43.10           | 83.51               | 67.47             | 68.74         |
| MGeo社区版       | **92.39**        | **79.99**    | **54.18**       | **86.09**           | **70.19**         | **79.55**     |

## 直接使用 MGeo训好的下游模型 来做推理
为了推动地址处理技术社区发展，我们提炼了常用的地址处理任务并建立了地理语义理解能力评测基准GeoGLUE。我们使用MGeo底座在GeoGLUE中提供的任务数据集上进行了训练，训练效果见上一部分**模型效果**。训好的模型目前提供一键试用。

### 门址地址要素解析

常见的门址地址作为寄递或外卖服务中用户位置的表达，一般包含以下几类信息：
- 行政区划信息，如省、市、县、乡镇信息;
- 路网信息，如路名，路号，道路设施等;
- 详细地理信息，如POI (兴趣点)、楼栋号、户室号等;
- 非地理信息，如补充说明，误输入等;
本任务需要对将门址地址拆分成独立语义的要素，并对这些要素进行类型识别。

基于MGeo训练好的模型已提供一键使用，详情见ModelCard : [点我试用训练好的门址地址要素解析模型](https://modelscope.cn/models/damo/mgeo_geographic_elements_tagging_chinese_base/summary)

### 地理实体对齐

日常生活中输入的地理文本可以为以下几种形式：
- 包含四级行政区划及路名路号POI的规范地理文本；
- 要素缺省的规范地理文本，例：只有路名+路号、只有POI；
- 非规范的地理文本、口语化的地理信息描述，例：阿里西溪园区东门旁亲橙里；
每一种地理文本的表述都可以映射到现实地理世界的道路、村庄、POI等实体上。地理实体对齐任务需要判断两段地理文本是否指代同一地理实体。该任务是构建地理信息知识库的核心技术。

基于MGeo训练好的模型已提供一键使用，详情见ModelCard : [点我试用训练好的地理实体对齐模型](https://modelscope.cn/models/damo/mgeo_geographic_entity_alignment_chinese_base/summary)

### Query-POI相关性排序

POI（Point Of Interest，兴趣点）搜索是地图类应用的核心功能，需要根据用户的query找到对应地图上的POI。一个POI的组成通常是其对应的地址描述以及经纬度描述。本任务需要在给定用户query以及候选POI列表的情况下，根据POI与query的相关性程度对POI进行排序。

基于MGeo训练好的模型已提供一键使用，详情见ModelCard : [点我试用训练好的Query-POI相关性排序](https://modelscope.cn/models/damo/mgeo_geographic_textual_similarity_rerank_chinese_base/summary)

### 地址Query成分分析

用户在地图类服务中输入地址query用以查询相关地理元素信息。地址query除了包含标准门址信息外通常还包含：
- 公交、地铁站点、线路信息；
- 品牌、商圈名称；
- 修饰信息如：24小时，便宜的；
- 语义连接词如： 助词、疑问词、连接词、转折词、需求词、附近词、量词。
相比标准门址，地址query的表达更丰富，信息量更大。本任务需要对地址query进行成分分析。

基于MGeo训练好的模型已提供一键使用，详情见ModelCard : [点我试用训练好的地址Query成分分析模型](https://modelscope.cn/models/damo/mgeo_geographic_composition_analysis_chinese_base/summary)

### WhereWhat切分

当一条地址包含多个地点描述时，通常需要对其进行切分，将原始地址切为where和what两部分。例如：
三里屯酒吧
这条描述，包含两个部分：三里屯、酒吧。用户的意图是想找三里屯附近的酒吧。因此需要将原始地址的“三里屯”识别为where部分，“酒吧”识别为what部分。
本任务目标是将输入地址做where和what的切分与识别。

基于MGeo训练好的模型已提供一键使用，详情见ModelCard : [点我试用训练好的WhereWhat切分模型](https://modelscope.cn/models/damo/mgeo_geographic_where_what_cut_chinese_base/summary)


## 自定义训练

当用户有自己标注好的数据希望基于MGeo底座进行训练时，可使用自定义训练功能。我们针对序列标注、分类、排序三类任务提供示例代码。

### 序列标注
以GeoGLUE的门址地址要素解析任务为例
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
    cfg.task = 'token-classification'
    cfg['dataset'] = {
        'train': {
            'labels': label_enumerate_values,
            'first_sequence': 'tokens',
            'label': 'ner_tags',
            'sequence_length': 128
        }
    }
    cfg['preprocessor'] = {
        'type': 'token-cls-tokenizer',
        'padding': 'max_length'
    }
    cfg.train.max_epochs = 1
    cfg.train.dataloader.batch_size_per_gpu = 32
    cfg.train.optimizer.lr = 3e-5
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

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

# load dataset
train_dataset = MsDataset.load('GeoGLUE', subset_name='GeoETA', split='train', namespace='damo')
dev_dataset = MsDataset.load('GeoGLUE', subset_name='GeoETA', split='validation', namespace='damo')

label_enumerate_values = get_label_list(train_dataset._hf_ds['train']['ner_tags'] + dev_dataset._hf_ds['validation']['ner_tags'])


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
{"tokens": ["浙", "江", "杭", "州", "市", "江", "干", "区", "九", "堡", "镇", "三", "村", "村", "一", "区"], "ner_tags": ["B-prov", "E-prov", "B-city", "I-city", "E-city", "B-district", "I-district", "E-district", "B-town", "I-town", "E-town", "B-community", "I-community", "E-community", "B-poi", "E-poi"]}
```

然后替换原流程中的train_dataset和dev_dataset为：
```python
local_train = 'train.json'
local_test = 'dev.json'
train_dataset = MsDataset.load('json', data_files={'train': [local_train]})
dev_dataset = MsDataset.load('json', data_files={'validation': [local_test]})
```

### 分类任务
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

### 排序任务
以GeoGLUE的Query-POI排序任务为例
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
    neg_sample = 19
    cfg.task = 'text-ranking'
    cfg['preprocessor'] = {'type': 'mgeo-ranking'}
    cfg.train.optimizer.lr = 5e-5
    cfg['dataset'] = {

        'train': {
            'type': 'mgeo',
            'query_sequence': 'query',
            'pos_sequence': 'positive_passages',
            'neg_sequence': 'negative_passages',
            'text_fileds': ['text', 'gis'],
            'qid_field': 'query_id',
            'neg_sample': neg_sample,
            'sequence_length': 64
        },
        'val': {
            'type': 'mgeo',
            'query_sequence': 'query',
            'pos_sequence': 'positive_passages',
            'neg_sequence': 'negative_passages',
            'text_fileds': ['text', 'gis'],
            'qid_field': 'query_id'
        },
    }
    cfg.evaluation.dataloader.batch_size_per_gpu = 16
    cfg.train.dataloader.batch_size_per_gpu = 3
    cfg.train.dataloader.workers_per_gpu = 16
    cfg.evaluation.dataloader.workers_per_gpu = 16

    cfg['evaluation']['metrics'] = "mrr@1"
    cfg.train.max_epochs = 1
    cfg.model['neg_sample'] = neg_sample
    cfg.model['gis_num'] = 2
    cfg.model['finetune_mode'] = 'multi-modal'
    cfg.train.hooks = [{
        'type': 'CheckpointHook',
        'interval': 1
    }, {
        'type': 'TextLoggerHook',
        'interval': 100
    }, {
        'type': 'IterTimerHook'
    }, {
        'type': 'EvaluationHook',
        'by_epoch': True
    }]
    # lr_scheduler的配置

    cfg.train.lr_scheduler = {
            "type": "LinearLR",
            "start_factor": 1.0,
            "end_factor": 0.5,
            "total_iters": int(len(train_ds) / cfg.train.dataloader.batch_size_per_gpu) * cfg.train.max_epochs,
            "options": {
                "warmup": {
                    "type": "LinearWarmup",
                    "warmup_iters": int(len(train_ds) / cfg.train.dataloader.batch_size_per_gpu)
                },
                "by_epoch": False
            }
        }

    return cfg

# load dataset
train_dataset = MsDataset.load('GeoGLUE', subset_name='GeoTES-rerank', split='train', namespace='damo')
dev_dataset = MsDataset.load('GeoGLUE', subset_name='GeoTES-rerank', split='validation', namespace='damo')

train_ds = train_dataset['train']
dev_ds = dev_dataset['validation']

model_id = 'damo/mgeo_backbone_chinese_base'
finetune(
    model_id=model_id,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    cfg_modify_fn=cfg_modify_fn,
    name=Trainers.mgeo_ranking_trainer)

output_dir = os.path.join(tmp_dir, ModelFile.TRAIN_OUTPUT_DIR)
print(f'model is saved to {output_dir}')
```

如果需要从本地load用户自定义数据，可以先将数据处理为如下格式，并保存为train.json和dev.json：
```json
{"query_id": 0, "query": "丽华公寓(通惠中路)向北检验检疫科学研究所", "query_gis": "[[], [], [], [], [], \"120.59443087451544,30.315515932852602\"]", "idx": "0", "positive_passages": [{"text": "杭州中新街(惠港二路)76饶平县检验检疫局", "gis": "[[], [], [], [], [], \"120.20509044775532,30.076259797983873\"]"}], "negative_passages": [{"text": "杭州中新街", "gis": "[[], [], [], [], [], \"120.20509044775532,30.076259797983873\"]"}]}
```
例子中的gis信息为空缺状态，用户可补充上自己的gis信息，模型在有gis和无gis的场景下均可以进行推断，gis信息的加入可以提升模型效果。更新原始代码中的neg_sample数量为自定义训练集的负例个数。


然后替换原流程中的train_ds和dev_ds为：
```python
local_train = 'train.json'
local_test = 'dev.json'
train_dataset = MsDataset.load('json', data_files={'train': [local_train]})
dev_dataset = MsDataset.load('json', data_files={'validation': [local_test]})
train_ds = train_dataset.to_hf_dataset()
dev_ds = dev_dataset.to_hf_dataset()
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