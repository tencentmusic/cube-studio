

# SiameseUIE通用信息抽取模型介绍

SiameseUIE通用信息抽取模型，基于提示（Prompt）+文本（Text）的构建思路，利用指针网络（Pointer Network）实现片段抽取（Span Extraction），从而实现命名实体识别（NER）、关系抽取（RE）、事件抽取（EE）、属性情感抽取（ABSA）等多类任务的抽取。和市面上已有的通用信息抽取模型不同的是：

- **更通用**：SiameseUIE基于递归的训练推理架构，不仅可以实现常见的NER、RE、EE、ABSA这类包含一个或两个抽取片段的信息抽取任务，也可以实现包含更多抽取片段的信息抽取任务，例如：比较观点识别五元组任务等等，因此更为通用，几乎可以解决所有的信息抽取问题；
- **更高效**：SiameseUIE基于孪生神经网络的思想，将预训练语言模型（PLM）的前 N-n 层改为双流，后 n 层改为单流。我们认为语言模型的底层更多的是实现局部的简单语义信息的交互，顶层更多的是深层信息的交互，因此前N-n层不需要让Prompt和Text做过多的交互，我们将前N-n层Text的隐向量表示缓存了下来，从而将**推理速度提升了30%**；
- **更精准**：我们在4类任务、6个领域、9个数据集上进行了测试，**在零样本情况下，F1 Score较竞品模型提升24.6%，在少样本情况下(去除了部分数量较少的数据集)提升3-5个百分点**；

## 模型描述

模型基于structbert-base-chinese在千万级远监督数据+有监督数据预训练得到，模型框架如下图：

![](model.jpg)

## 期望模型使用方式以及适用范围
你可以使用该模型，实现命名实体识别（NER）、关系抽取（RE）、事件抽取（EE）、属性情感抽取（ABSA）等各类信息抽取任务。

### 如何使用

#### 安装Modelscope

依据ModelScope的介绍，实验环境可分为两种情况。在此推荐使用第2种方式，点开就能用，省去本地安装环境的麻烦，直接体验ModelScope。

##### 1 本地环境安装

可参考[ModelScope环境安装](https://www.modelscope.cn/?spm=a2c6h.12873639.article-detail.7.59b93bc77Qw9sE#/docs/环境安装)。

##### 2 Notebook

ModelScope直接集成了线上开发环境，用户可以直接在线训练、调用模型。

打开模型页面，点击右上角“在Notebook中打开”，选择机器型号后，即可进入线上开发环境。

#### 代码范例
##### Fine-Tune 微调示例
```python

import os
import json
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import read_config
from modelscope.metainfo import Metrics
from modelscope.utils.constant import DownloadMode


model_id = 'damo/nlp_structbert_siamese-uie_chinese-base'

WORK_DIR = '/tmp'

train_dataset = MsDataset.load('people_daily_ner_1998_tiny', namespace='damo', split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD)
eval_dataset = MsDataset.load('people_daily_ner_1998_tiny', namespace='damo', split='validation', download_mode=DownloadMode.FORCE_REDOWNLOAD)


max_epochs=3
kwargs = dict(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_epochs=max_epochs,
    work_dir=WORK_DIR)


trainer = build_trainer('siamese-uie-trainer', default_args=kwargs)

print('===============================================================')
print('pre-trained model loaded, training started:')
print('===============================================================')

trainer.train()

print('===============================================================')
print('train success.')
print('===============================================================')

for i in range(max_epochs):
    eval_results = trainer.evaluate(f'{WORK_DIR}/epoch_{i+1}.pth')
    print(f'epoch {i} evaluation result:')
    print(eval_results)


print('===============================================================')
print('evaluate success')
print('===============================================================')
```

##### 零样本推理示例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.siamese_uie, 'damo/nlp_structbert_siamese-uie_chinese-base', model_revision='v1.0')

# 命名实体识别 {实体类型: None}
semantic_cls(
    input='1944年毕业于北大的名古屋铁道会长谷口清太郎等人在日本积极筹资，共筹款2.7亿日元，参加捐款的日本企业有69家。', 
    schema={
        '人物': None,
        '地理位置': None,
        '组织机构': None
    }
) 
# 关系抽取 {主语实体类型: {关系(宾语实体类型): None}}
semantic_cls(
	input='在北京冬奥会自由式中，2月8日上午，滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌。2月9日上午，滑雪男子大跳台决赛中日本选手小泉次郎以188.25分获得银牌！', 
  	schema={
        '人物': {
            '比赛项目(赛事名称)': None,
            '参赛地点(城市)': None,
            '获奖时间(时间)': None,
            '选手国籍(国籍)': None
        }
    }
) 

# 事件抽取 {事件类型（事件触发词）: {参数类型: None}}
semantic_cls(
	input='7月28日，天津泰达在德比战中以0-1负于天津天海。', 
  	schema={
        '胜负(事件触发词)': {
            '时间': None,
            '败者': None,
            '胜者': None,
            '赛事名称': None
        }
    }
) 

# 属性情感抽取 {属性词: {情感词: None}}
semantic_cls(
	input='很满意，音质很好，发货速度快，值得购买', 
  	schema={
        '属性词': {
            '情感词': None,
        }
    }
) 
# 允许属性词缺省，#表示缺省
semantic_cls(
	input='#很满意，音质很好，发货速度快，值得购买', 
  	schema={
        '属性词': {
            '情感词': None,
        }
    }
) 
# 支持情感分类
semantic_cls(
	input='很满意，音质很好，发货速度快，值得购买', 
  	schema={
        '属性词': {
            "正向情感(情感词)": None, 
            "负向情感(情感词)": None, 
            "中性情感(情感词)": None}
        }
    }
) 
```
### 模型局限性以及可能的偏差
模型在较冷门的场景，效果可能不及预期。

## 数据评估及结果
我们在4类任务、6个领域、9个数据集上进行了测试，我们选择[DuUIE](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie)作为Baseline，在零样本情况下，**F1 Score较DuUIE模型提升24.6%；**
![](zeroshot.jpg)

在少样本情况下(去除了部分数量较少无法微调的数据集)**F1 Score较竞品模型提升3-5个百分点**；
![](fewshot.jpg)

### 相关论文以及引用信息
```bib
@article{wang2019structbert,
  title={Structbert: Incorporating language structures into pre-training for deep language understanding},
  author={Wang, Wei and Bi, Bin and Yan, Ming and Wu, Chen and Bao, Zuyi and Xia, Jiangnan and Peng, Liwei and Si, Luo},
  journal={arXiv preprint arXiv:1908.04577},
  year={2019}
}
@inproceedings{Zhao2021AdjacencyLO,
  title={Adjacency List Oriented Relational Fact Extraction via Adaptive Multi-task Learning},
  author={Fubang Zhao and Zhuoren Jiang and Yangyang Kang and Changlong Sun and Xiaozhong Liu},
  booktitle={FINDINGS},
  year={2021}
}
```