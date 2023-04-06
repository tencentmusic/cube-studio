
# SPACE-T表格问答中文大规模预训练模型介绍

中文表格问答（TableQA）模型是一个多轮表格知识预训练语言模型，可用于解决下游的多轮Text-to-SQL语义解析任务。该模型并通过海量中文表格数据预训练(千万级)，在中文Text2SQL数据集上取得不错的效果。本模型是SPACE系列模型的一员，SPACE-T（SPACE-Table的简称），SPACE系列的其他模型可参考[SPACE预训练对话模型](https://www.modelscope.cn/models/damo/nlp_space_pretrained-dialog-model/)。团队其他研究进展可以参考[DAMO-ConvAI](https://github.com/AlibabaResearch/DAMO-ConvAI)。

任务简要描述：给定表格（例如基金信息表）的情况下，用户输入基于表格的自然语言问题（例如，XX公司的基金有哪些风险类型？），模型会将用户的自然语言问题转化成SQL查询语句（例如，SELECT 风险类型 FROM 基金信息表 WHERE 公司名称 = XX），并且在该表格对应的数据库中执行该SQL语句，从而返回查询结果（例如，“低风险”、“中高风险”）；（详细示例见代码范例）

## 模型描述

中文Text2SQL大规模预训练模型采用大规模的中文表格进行预训练，并且在Text2SQL数据集上进行微调，使得模型具备理解各种领域的表格知识的基础能力，可用于解决下游的多轮 Text-to-SQL 语义解析任务。

模型结构上，采用统一的 Transformer 架构作为模型底座，对输入的自然语言问题和表格的schema结构进行理解。然后，采用sketch-based方法分别预测SQL语句中select子句和where子句，从而构成最终的SQL语句。模型结构如下图：

<p align="center">
    <img src="resources/star.jpg" alt="star" width="800"/>
</p>

## DEMO示例
本模型有对应的DEMO示例已经上线，但是由于这里的DEMO界面较为简单无法展现历史对话，因此DEMO只支持单轮问答的结果，如果用户想体验多轮对话的效果，可以通过代码范例里的逻辑进行调用。

另外，SPACE-T表格问答支持同比、环比、折线/柱状图等展示的功能，已经在[表格BI问答](https://modelscope.cn/studios/damo/Text2SQL4BI/)中上线，感兴趣可以跳转到此进行试用和进一步开发。

## 模型能力描述

### 基础能力表格

基础能力指的是模型具备的一些原子能力，可以通过右侧的在线体验直接进行测试这些能力。

| 能力 | 解释 | 示例问题 | 示例SQL |
| ---------- | ---------- | ---------- | ---------- |
| 多列查询 | SELECT子句中支持选择多个不同的column | 列出油耗大于8但是功率低于200的**名称和价格** | SELECT **产品名称, 零售价** FROM 汽车 WHERE ( 综合耗油量 > "8" ) AND ( 功率 < "200" ) |
| 聚合函数查询 | SELECT子句支持选择不同的聚合函数，包括：COUNT、SUM、AVG | 上个月收益超过3的**有几个**基金？ | SELECT **COUNT(基金名称)** FROM 基金 WHERE ( 月收益率 > "3" ) |
| | | 长江流域和珠江流域的水库**库容总量**是多少？ | SELECT **SUM(库容)** FROM 水库 WHERE ( 所在流域 == "长江" ) OR ( 所在流域 == "珠江" ) |
| 值比较条件 | WHERE子句支持等于、大于、小于、不等于运算符 | **计算机**或者**成绩优秀**的同学有哪些？学号是多少？ | SELECT 学号, 学位 FROM 学生信息 WHERE ( **专业名称 == "计算机"** ) OR ( **成绩 == "优秀"** ) |
| | | 列出**油耗大于8**但是**功率低于200**的名称和价格 | SELECT 产品名称, 零售价 FROM 汽车 WHERE ( **综合耗油量 > "8"** ) AND ( **功率 < "200"** ) |
| | | **净值不等于1**的基金平均月收益率和年收益率是多少？ | SELECT AVG(月收益率), AVG(今年年收益率) FROM 基金 WHERE ( **净值 != "1"** ) |
| 多条件并存 | WHERE子句支持多个条件以AND或OR的形式组合查询 | 长江流域**和**珠江流域的水库库容总量是多少？ | SELECT SUM(库容) FROM 水库 WHERE ( 所在流域 == "长江" ) **OR** ( 所在流域 == "珠江" ) |
| | | 列出油耗大于8**但是**功率低于200的名称和价格 | SELECT 产品名称, 零售价 FROM 汽车 WHERE ( 综合耗油量 > "8" ) **AND** ( 功率 < "200" ) |
| 自动补充列名 | 查询列名=值的情况下，用户可以省略列名 | **计算机**或者成绩优秀的同学有哪些？学号是多少？ | SELECT 学号, 学位 FROM 学生信息 WHERE ( **专业名称 == "计算机"** ) OR ( 成绩 == "优秀" ) |
| | | 油耗低于5的**suv**有哪些？ | SELECT 产品名称 FROM 汽车 WHERE ( **汽车类型 == "suv"** ) AND ( 综合耗油量 < "5" ) |
| 一定的泛化能力 | 对于列名的询问不要求完全匹配表格中的列名 | **油耗**低于5的suv有哪些？ | SELECT 产品名称 FROM 汽车 WHERE ( 汽车类型 == "suv" ) AND ( **综合耗油量** < "5" ) |
| | | **上个月收益**超过3的有几个基金？ | SELECT COUNT(基金名称) FROM 基金 WHERE ( **月收益率** > "3" ) |
| 拒识能力 | 拒绝和表格无关的询问 | 今天星期几？ | SELECT 空列 |
| | | 冬至吃不吃饺子？ | SELECT 空列 |
| 多轮对话能力（SDK中可使用，在线体验DEMO中无法使用） | 记录历史信息并进行多轮对话 | 1. 珠江流域的小型水库的库容总量是多少 </br> 2. 那平均值是多少？ </br> 3. 换成中型的呢？ | 1. SELECT SUM(库容) FROM 水库 WHERE ( 工程规模 == "小型" ) AND ( 所在流域 == "珠江" )  </br>  2. SELECT AVG(库容) FROM 水库 WHERE ( 工程规模 == "小型" ) AND ( 所在流域 == "珠江" ) </br> 3. SELECT AVG(库容) FROM 水库 WHERE ( 工程规模 == "中型" ) AND ( 所在流域 == "珠江" ) |

### 组合能力表格

组合能力指的是对基础能力的组合，例如用户提出的问题可能包含模型的多种基础能力，在此情况下，模型也能正确预测。如下表示例：

| 能力 | 示例问题 | 示例SQL |
| ---------- | ---------- | ---------- |
| 多列查询 + 多条件并存 + 自动补充列名 | 计算机或者成绩优秀的同学有哪些？学号是多少？ | SELECT 学号, 学位 FROM 学生信息 WHERE ( 专业名称 == "计算机" ) OR ( 成绩 == "优秀" ) |
| 多条件并存 + 值比较条件 + 自动补充列名 + 泛化能力 | 油耗低于5的suv有哪些？ | SELECT 产品名称 FROM 汽车 WHERE ( 汽车类型 == "suv" ) AND ( 综合耗油量 < "5" ) |
| 聚合函数查询 + 值比较条件 + 泛化能力 | 上个月收益超过3的有几个基金？ | SELECT COUNT(基金名称) FROM 基金 WHERE ( 月收益率 > "3" ) |


## 如何使用

你可以直接通过一个流水线使用模型用于多轮Text-to-SQL语义解析任务：

### 代码范例

使用TableQA-中文-通用领域-base模型需要安装 modelscope，安装方法在文档中心里可以找到。安装完成后，运行如下代码即可进行模型预测。

```python
import os, json
from transformers import BertTokenizer
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.preprocessors import TableQuestionAnsweringPreprocessor
from modelscope.preprocessors.nlp.space_T_cn.fields.database import Database
from modelscope.utils.constant import ModelFile, Tasks

model_id = 'damo/nlp_convai_text2sql_pretrain_cn'
test_case = {
    'utterance':
    [['长江流域的小型水库的库容总量是多少？', 'reservoir'], ['那平均值是多少？', 'reservoir'], ['那水库的名称呢？', 'reservoir'], ['换成中型的呢？', 'reservoir']]
}

model = Model.from_pretrained(model_id)
tokenizer = BertTokenizer(
    os.path.join(model.model_dir, ModelFile.VOCAB_FILE))
db = Database(
    tokenizer=tokenizer,
    table_file_path=os.path.join(model.model_dir, 'table.json'),
    syn_dict_file_path=os.path.join(model.model_dir, 'synonym.txt'),
    is_use_sqlite=True)
preprocessor = TableQuestionAnsweringPreprocessor(
    model_dir=model.model_dir, db=db)
pipelines = [
    pipeline(
        Tasks.table_question_answering,
        model=model,
        preprocessor=preprocessor,
        db=db)
]

for pipeline in pipelines:
    historical_queries = None
    for question, table_id in test_case['utterance']:
        output_dict = pipeline({
            'question': question,
            'table_id': table_id,
            'history_sql': historical_queries
        })[OutputKeys.OUTPUT]
        print('question', question)
        print('sql text:', output_dict[OutputKeys.SQL_STRING])
        print('sql query:', output_dict[OutputKeys.SQL_QUERY])
        print()
        historical_queries = output_dict[OutputKeys.HISTORY]
```

### 训练范例

调用fine-tuning流程可以参考下方的代码，依照情况调整超参数中的batch_size、total_epoches等。

```python
import os, json
from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.table_question_answering_trainer import TableQuestionAnsweringTrainer
from modelscope.utils.constant import DownloadMode

input_dataset = MsDataset.load(
    'ChineseText2SQL', download_mode=DownloadMode.FORCE_REDOWNLOAD)
train_dataset = []
for name in input_dataset['train']._hf_ds.data[1]:
    train_dataset.append(json.load(open(str(name), 'r')))
eval_dataset = []
for name in input_dataset['test']._hf_ds.data[1]:
    eval_dataset.append(json.load(open(str(name), 'r')))
print('size of training set', len(train_dataset))
print('size of evaluation set', len(eval_dataset))

model_id = 'damo/nlp_convai_text2sql_pretrain_cn'
trainer = TableQuestionAnsweringTrainer(
    model=model_id,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train(
    batch_size=8,
    total_epoches=2,
)
trainer.evaluate(
    checkpoint_path=os.path.join(trainer.model.model_dir, 'finetuned_model.bin'))
```

### 对话样例

| user question （输入）                                                                        | sql query （输出）                                                                                    |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| 长江流域的小型水库的库容总量是多少？ | SELECT SUM(`库容`) FROM `水库信息表` WHERE `所在流域` = `长江流域` AND `类型` = `小型` |
| 那平均值是多少？ | SELECT AVG(`库容`) FROM `水库信息表` WHERE `所在流域` = `长江流域` AND `类型` = `小型` |
| 那水库的名称呢？ | SELECT `水库名称` FROM `水库信息表` WHERE `所在流域` = `长江流域` AND `类型` = `小型` |
| 换成中型的呢？ | SELECT `水库名称` FROM `水库信息表` WHERE `所在流域` = `长江流域` AND `类型` = `中型` |

## 模型局限性以及可能的偏差
本项目支持用户使用本地自定义的的数据库进行预测，虽然本模型基于千万级且来源于各个领域的中文表格数据进行预训练，但是可能某些非常特殊的表格效果无法达到最优。另外，由于下游任务是都是限定领域的对话，不宜像使用闲聊对话系统一样进行开放域测试。

## 测试集及评估结果

测试TableQA-中文-通用领域-base模型可以使用[通用领域Text2SQL训练集](https://modelscope.cn/datasets/modelscope/ChineseText2SQL)中的testset。评估结果如下（由于ChineseText2SQL数据集只提供了100条测试数据，因此测试结果并不完全一样）：

| 测试集样本数 | **SQL exact match 准确率** | SELECT 列名 准确率 | SELECT 聚合函数 准确率 | WHERE 列名 准确率 | WHERE 操作符 准确率 | WHERE 值 准确率 |
| --------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| 1557 | **0.861** | 0.953 | 0.953 | 0.921 | 0.955 | 0.966 |

## 相关论文及引用
本文使用到的模型参考自团队内部的一些论文：

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