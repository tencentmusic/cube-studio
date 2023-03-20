

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

用户在地图类服务中输入地址query用以查询相关地理元素信息。地址query除了包含标准门址信息外通常还包含：
- 公交、地铁站点、线路信息；
- 品牌、商圈名称；
- 修饰信息如：24小时，便宜的；
- 语义连接词如： 助词、疑问词、连接词、转折词、需求词、附近词、量词。


相比标准门址，地址query的表达更丰富，信息量更大。



本任务需要对地址query进行成分分析，其输入输出如下：

- 输入：地址query
- 输出：query中包含的成分片段及类型


## 模型描述
地址由于其丰富的表达以及与地图联动的多模态属性，一直是自动化处理的一个难题。达摩院联合高德发布了多任务多模态地址预训练底座MGeo模型。该模型基于地图-文本多模态架构，使用多任务预训练（MOMETAS）技术融合了注意力对抗预训练（ASA）、句子对预训练（MaSTS）、多模态预训练，训练得到适合于多类地址任务的预训练底座，为下游广泛的地址处理任务带来性能提升。
- MOMETAS：动态融合多种预训练目标得到普适性更强的预训练底座，技术发表于EMNLP2022（[论文链接](https://arxiv.org/abs/2210.10293)）。
- ASA：对预训练时的自注意力进行对抗攻击训练避免对局部信息的过多关注，技术发表于AAAI2023（[论文链接](https://arxiv.org/abs/2206.12608)）。
- MaSTS：更适合于捕捉句子对之间关系的预训练技术，登顶CLUE语义匹配榜首，通用版模型已开源（[模型链接](https://www.modelscope.cn/models/damo/nlp_masts_sentence-similarity_clue_chinese-large/summary)）。
- 地图-文本多模态预训练：业内首次实现对于地图的建模表示以及地图-文本的跨模态融合（[论文链接](https://arxiv.org/abs/2301.04283)）。

在ModelScope中我们开源的版本是基于开源地址数据以及开源地图OpenStreetMap训练的MGeo预训练底座以及其在GeoGLUE地理语义评测榜中多个任务的下游模型。

## 标签说明
| Label                            | Name                                                                                            |
|----------------------------------|-------------------------------------------------------------------------------------------------|
| BS                            | 公交地铁站                                                                                      |
| BL                            | 公交地铁线路                                                                                    |
| RD                            | 道路、(高速公路、阜荣街)、隧道、桥梁、立交桥                                                    |
| Entity                        | POI一般名称:黄山风景区、首开广场、清华大学、(幼儿园)春蕾                                        |
| Brand                         | 著名品牌:((阜荣街)外婆家)                                                                       |
| CategorySuffix                | 类别后缀词:(阜荣街)大学（二层)                                                                  |
| PA                            | 国家                                                                                            |
| PB                            | 省                                                                                              |
| PC                            | 城市                                                                                            |
| PD                            | 区县(包括开发区)                                                                                |
| PE                            | 乡镇                                                                                            |
| PF                            | 街道                                                                                            |
| PG                            | 村庄                                                                                            |
| PH                            | 行政俗称/商圈：九亭、古荡、莘庄、望京                                                           |
| PS                            | 其它行政                                                                                        |
| UA                            | 门址 道路xx号/xx弄                                                                              |
| UB                            | 门址 xx座楼/xx区                                                                                |
| UC                            | 门址 xx号楼/xx棟/xx幢/                                                                          |
| UD                            | 门址 附属描述(三期、北段、一段、(机场)出发、内环、外环、路口、交叉口、下道口、交汇处、高速出口) |
| UE                            | 门址 东口、南门                                                                               |
| Desc  |修饰词：24小时，便宜的                                                                                                 |
| Yewu                    |业务词                                                                                                 |
| SS                            | 分支词：(肯德基)xx店、(xx)紫金港校区、(xx)望京支行、(xx)二厂                                    |
| SA                            | 方位修饰词；南，向南，东侧、路南、桥北                                                          |
| YA                            | 语义连接词: 助词、疑问词、连接词、转折词、需求词、附近词、量词                                  |
| BD                            | 标点符号                                                                                        |
| NumEng                        | 数字英文串(品牌、entity、明确的门址除外)                                                        |
| ZZ                            | 未知                                                                                            |


## 一键调用
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

task = Tasks.token_classification
model = 'damo/mgeo_geographic_composition_analysis_chinese_base'
inputs = '浙江省杭州市余杭区阿里巴巴西溪园区'
pipeline_ins = pipeline(
    task=task, model=model)
print(pipeline_ins(input=inputs))
# 输出
# {'output': [{'type': 'PB', 'start': 0, 'end': 3, 'span': '浙江省'}, {'type': 'PC', 'start': 3, 'end': 6, 'span': '杭州市'}, {'type': 'PD', 'start': 6, 'end': 9, 'span': '余杭区'}, {'type': 'Entity', 'start': 9, 'end': 17, 'span': '阿里巴巴西溪园区'}]}
```

## 自定义训练

当用户有自己标注好的数据希望基于MGeo底座进行训练或基于训练好的下游模型进行继续训练时，可使用自定义训练功能。


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