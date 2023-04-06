
# SPACE-T多轮表格知识预训练语言模型介绍
该模型是一个多轮表格知识预训练语言模型，可用于解决下游的多轮Text-to-SQL语义解析任务。模型采用基于模板和回译方法生成的全小写英文合成语料进行预训练。

本项目的模型是基于一个多轮Text-to-SQL数据集 CoSQL 微调后的下游模型，可针对不同领域数据库和用户直接进行多轮对话，生成相应的SQL查询语句。用户可以在对话过程中表达自己对数据库模式的查询要求，并在系统的帮助下生成符合要求的SQL查询语句。


## 模型描述
本模型采用人工构建的多轮Text-to-SQL预训练数据进行预训练，采用统一的单个Transformer架构作为模型底座。
模型采用半监督的方式在多轮Text-to-SQL预训练数据上进行训练，采用3个预训练目标：模式状态追踪建模，对话依赖追踪建模和掩码语言建模，最后以多任务学习的方式进行训练。
<p align="center">
    <img src="star2.png" alt="donuts" width="400" />

在下游任务Text-to-SQL微调的时候，我们直接利用本模型作为底座，基于常用的下游模型 [lgesql](https://github.com/rhythmcao) 进行微调得到。 具体做法就是将 lgesql 的 ELECTRA 底座替换成本模型, 并修改输入格式，其余保持不变，效果上即可得到显著提升。


## 期望模型使用方式以及适用范围
你可以使用本模型针对任意领域进行对话。
输入用户语句和相应的数据库，模型就能够开始多轮交互，帮助用户生成当前对话相应的SQL查询语句。

### 如何使用

你可以直接通过一个流水线使用模型用于多轮Text-to-SQL语义解析任务，在notebook中选择V100 GPU环境：

#### 代码范例
```
pip install text2sql_lgesql  -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.preprocessors import ConversationalTextToSqlPreprocessor
from modelscope.utils.constant import Tasks

model_id = 'damo/nlp_star_conversational-text-to-sql'
test_case = {
    "database_id": 'employee_hire_evaluation',
    'local_db_path':None,
    "utterance":[
    "I'd like to see Shop names.",
    "Which of these are hiring?",
    "Which shop is hiring the highest number of employees? | do you want the name of the shop ? | Yes"]
}

model = Model.from_pretrained(model_id)
preprocessor = ConversationalTextToSqlPreprocessor(model_dir=model.model_dir)
pipeline = pipeline(
                    task=Tasks.table_question_answering,
                    model=model,
                    preprocessor=preprocessor)
last_sql, history = '', []
for item in test_case['utterance']:
    case = {"utterance": item,
            "history": history,
            "last_sql": last_sql,
            "database_id": test_case['database_id'],
            'local_db_path': test_case['local_db_path']}
    results = pipeline(case)
    print(results)
    history.append(item)
```
> **NOTE**:本项目支持用户使用本地自定义的的数据库，请仿照db文件和tables.json文件设置数据库格式，并传入正确的地址'local_db_path'。

#### 对话样例

| user utterance （输入）                                                                        | system response （输出）                                                                                    |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| I'd like to see Shop names. | SELECT shop.Name FROM shop |
| Which of these are hiring? | SELECT shop.Name FROM shop JOIN hiring |
| Which shop is hiring the highest number of employees? \| do you want the name of the shop ? \| Yes | SELECT shop.Name FROM hiring JOIN shop GROUP BY hiring.Shop_ID ORDER BY COUNT(*) DESC LIMIT 1 |

### 模型局限性以及可能的偏差
本模型基于CoSQL数据集(多轮Text-to-sql数据集)训练，训练数据有限，效果可能存在一定偏差。由于下游任务是都是限定领域的对话，不宜像使用闲聊对话系统一样进行开放域测试。本项目支持用户使用本地自定义的的数据库，但由于训练数据有限，效果可能存在较大偏差，请用户自行评测后决定如何使用.

## 训练数据介绍
CoSQL 是一个跨领域的多轮Text-to-SQL数据，是多轮Text-to-SQL最受欢迎的 benchmark，一共包括了1.5万个Text-to-SQL sessions。详见[CoSQL](https://modelscope.cn/datasets/yuchen/CoSQL/summary).


## 数据评估及结果
模型进行多轮Text-to-SQL语义解析任务，在[CoSQL](https://modelscope.cn/datasets/yuchen/CoSQL/summary)数据集上取得了 SOTA 指标：

### Task：多轮Text-to-SQL解析任务 [CoSQL](https://modelscope.cn/datasets/yuchen/CoSQL/summary)

| Dataset Name   | Question Match (Dev)  |  Question Match (Test)  | Interaction Match (Dev) | Interaction Match (Test) | 
|:--------------:|:---------------------:|:-----------------------:|:-----------------------:|:------------------------:|
|     CoSQL      |         59.7          |           57.8          |          30.0           |            28.2          |

> **NOTE**: Question Match 表示所有问题的SQL查询语句的精确匹配度，Interaction Match 表示所有正确预测的问题的交互率。

## 相关论文及引用
本文使用到的模型参考自团队内部的一些论文，相关代码可以参考团队[代码仓库](https://github.com/AlibabaResearch/DAMO-ConvAI)：

```
@inproceedings{cai2022star, 
    title={STAR: SQL Guided Pre-Training for Context-dependent Text-to-SQL Parsing}, 
    author={Cai, Zefeng and Li, Xiangyu and Hui, Binyuan and Yang, Min and Li, Bowen and Li, Binhua and Cao, Zheng and Li, Weijie and Huang, Fei and Si, Luo and Li, Yongbin}, 
    booktitle={EMNLP}, 
    year={2022} 
}
@inproceedings{he2022unified, 
    title={Unified Dialog Model Pre-training for Task-Oriented Dialog Understanding and Generation}, 
    author={He, Wanwei and Dai, Yinpei and Yang, Min and Huang, Fei and Si, Luo and Sun, jian and Li, Yongbin}, 
    booktitle={SIGIR}, 
    year={2022} 
}
```