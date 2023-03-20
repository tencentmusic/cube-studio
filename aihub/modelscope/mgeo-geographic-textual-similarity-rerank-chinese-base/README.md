
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

POI（Point Of Interest，兴趣点）搜索是地图类应用的核心功能，需要根据用户的query找到对应地图上的POI。一个POI的组成通常是其对应的地址描述以及经纬度描述。本任务需要在给定用户query以及候选POI列表的情况下，根据POI与query的相关性程度对POI进行排序。


任务的每一条输入包括用户query、用户位置以及候选POI列表，每个POI包括POI的地址描述以及POI位置。需要根据query与POI的相关性按照相关度从高到底为POI打分。



输入：用户query、用户位置、候选POI列表


输出：POI得分


注意本模型基于OpenStreetMap杭州POI进行训练，不保证在其他POI库以及其他地方query上的效果。


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

task = Tasks.text_ranking
model = 'damo/mgeo_geographic_textual_similarity_rerank_chinese_base'
# 多模态输入，包括需要排序的文本以及地理信息
multi_modal_inputs = {
"source_sentence": ['杭州余杭东方未来学校附近世纪华联商场(金家渡北苑店)'],
"first_sequence_gis": [[[13159, 13295, 13136, 13157, 13158, 13291, 13294, 74505, 74713, 75387, 75389, 75411], [3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4], [3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [[1254, 1474, 1255, 1476], [1253, 1473, 1256, 1476], [1247, 1473, 1255, 1480], [1252, 1475, 1253, 1476], [1253, 1475, 1253, 1476], [1252, 1471, 1254, 1475], [1254, 1473, 1256, 1475], [1238, 1427, 1339, 1490], [1238, 1427, 1339, 1490], [1252, 1474, 1255, 1476], [1252, 1474, 1255, 1476], [1249, 1472, 1255, 1479]], [[24, 23, 15, 23], [24, 28, 15, 18], [31, 24, 22, 22], [43, 13, 37, 13], [43, 6, 35, 6], [31, 32, 22, 14], [19, 30, 9, 16], [24, 30, 15, 16], [24, 30, 15, 16], [29, 24, 20, 22], [28, 25, 19, 21], [31, 26, 22, 20]], "120.08802231437534,30.343853313981505"]],
"sentences_to_compare": [
'良渚街道金家渡北苑42号世纪华联超市(金家渡北苑店)',
'金家渡路金家渡中苑南区70幢金家渡中苑70幢',
'金家渡路140-142号附近家家福足道(金家渡店)'
],
"second_sequence_gis": [
[[13083, 13081, 13084, 13085, 13131, 13134, 13136, 13147, 13148], [3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 4, 4, 4, 4, 4, 4, 4, 4], [[1248, 1477, 1250, 1479], [1248, 1475, 1250, 1476], [1247, 1478, 1249, 1481], [1249, 1479, 1249, 1480], [1249, 1476, 1250, 1476], [1250, 1474, 1252, 1478], [1247, 1473, 1255, 1480], [1250, 1478, 1251, 1479], [1249, 1478, 1250, 1481]], [[30, 26, 21, 20], [32, 43, 23, 43], [33, 23, 23, 23], [31, 13, 22, 13], [25, 43, 16, 43], [20, 33, 10, 33], [26, 29, 17, 17], [18, 21, 8, 21], [26, 23, 17, 23]], "120.08075205680345,30.34697777462197"],
[[13291, 13159, 13295, 74713, 75387, 75389, 75411], [3, 3, 3, 4, 4, 4, 4], [3, 4, 4, 4, 4, 4, 4], [[1252, 1471, 1254, 1475], [1254, 1474, 1255, 1476], [1253, 1473, 1256, 1476], [1238, 1427, 1339, 1490], [1252, 1474, 1255, 1476], [1252, 1474, 1255, 1476], [1249, 1472, 1255, 1479]], [[28, 28, 19, 18], [22, 16, 12, 16], [23, 24, 13, 22], [24, 30, 15, 16], [27, 20, 18, 20], [27, 21, 18, 21], [30, 24, 21, 22]], "120.0872539617001,30.342783672056953"],
[[13291, 13290, 13294, 13295, 13298], [3, 3, 3, 3, 3], [3, 4, 4, 4, 4], [[1252, 1471, 1254, 1475], [1253, 1469, 1255, 1472], [1254, 1473, 1256, 1475], [1253, 1473, 1256, 1476], [1255, 1467, 1258, 1472]], [[32, 25, 23, 21], [26, 33, 17, 33], [21, 19, 11, 19], [25, 21, 16, 21], [21, 33, 11, 33]], "120.08839673752281,30.34156156893651"]
]
}

# 单模态输入，只包括需要排序的文本
single_modal_inputs = {
"source_sentence": ['杭州余杭东方未来学校附近世纪华联商场(金家渡北苑店)'],
"sentences_to_compare": [
'良渚街道金家渡北苑42号世纪华联超市(金家渡北苑店)',
'金家渡路金家渡中苑南区70幢金家渡中苑70幢',
'金家渡路140-142号附近家家福足道(金家渡店)'
]
}

# 模型可接受多模态输入
pipeline_ins = pipeline(
    task=task, model=model)
print(pipeline_ins(input=multi_modal_inputs))
# 输出
# {'scores': [0.9997552633285522, 0.027718106284737587, 0.03500296175479889]}

# 模型可接受单模态输入
pipeline_ins = pipeline(
    task=task, model=model)
print(pipeline_ins(input=single_modal_inputs))
# 输出
# {'scores': [0.9986912608146667, 0.0075200702995061874, 0.014017169363796711]}
```

## 自定义训练

当用户有自己标注好的数据希望基于MGeo底座进行训练或基于训练好的下游模型进行继续训练时，可使用自定义训练功能。


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

    # 数据量较大，限制训练步数为10，如需全量训练可删去下面两句
    cfg.train.train_iters_per_epoch = 10
    cfg.evaluation.val_iters_per_epoch = 10

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


然后替换原流程中的train_dataset和dev_dataset为：
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