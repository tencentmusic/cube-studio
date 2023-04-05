
# SPACE 对话状态跟踪模型介绍

SPACE 模型是一个统一的半监督预训练对话模型，可用于解决下游各种类型的任务型对话任务。模型采用大量无标对话语料和少量对话标注知识进行预训练，统一的架构使得模型可同时用于对话理解和对话生成类任务。模型采用全小写英文语料进行训练和使用。

本项目的模型是 SPACE 基于一个对话状态跟踪数据集 MultiWOZ2.2 微调后的下游模型，称作 space_dialog-state-tracking，可专门用来做旅游、餐馆等领域的对话状态跟踪任务。


## 模型描述
SPACE同时采用有英文标注对话语料和英文无标注对话语料进行预训练，旨在对话预训练过程中融入对话标注知识。
模型采用统一的单个Transformer架构作为模型底座，由四个连续的组件组成，以建模任务型对话中的对话流：
- 对话编码模块：用于编码对话历史并且捕捉公共的对话上下文表征
- 对话理解模块：用于提取当前轮用户问题或者系统回复的语义向量
- 对话策略模块：用于生成代表当前轮系统回复的高层次语义的策略向量
- 对话生成模块：用于生成当前轮合适的系统回复语句以返回给用户

模型采用半监督的方式同时在有标数据和无标数据上进行训练，针对每个模型组件设计不同的预训练目标，最终模型采用5个预训练目标：片段掩码语言建模，半监督理解语义建模，语义区域建模，策略语义建模和对话生成建模，最后以多任务学习的方式进行训练。
因此，模型可以在预训练过程中同时学习到对话理解，对话策略和对话生成的能力，可以同时用于下游各种类型的任务型对话任务。
针对对话理解类任务，只需要复用模型的对话编码模块和对话理解模块进行编码；针对对话策略和对话生成类任务，需要使用完整模型的所有模块进行编码和生成。

在对话状态跟踪下游任务微调的时候，我们直接利用 SPACE 作为底座，基于一个大家常用的下游模型 [TripPy](https://aclanthology.org/2020.sigdial-1.4/) 进行微调得到。 具体做法就是将 TripPy 的 BERT 底座替换成 SPACE, 其余保持不变，效果上即可得到显著提升。 


## 期望模型使用方式以及适用范围
你可以使用 space_dialog-state-tracking 模型用于 MultiWOZ2.2 所有覆盖的领域的对话状态跟踪，领域包括 restaurant（餐馆），hotel（酒店），attraction（景点），train（火车）， taxi（计程车）。

输入完整的对话历史（包括用户语句、系统语句、系统动作），模型会给出当前轮的对话状态（即用户所表达出想要的槽值对）。

### 如何使用

你可以直接通过一个流水线使用模型用于对话状态追踪任务：

#### 代码范例

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

model_id = 'damo/nlp_space_dialog-state-tracking'
test_case = [
    {
        "User-1": "Hi, I\'m looking for a train that is going to cambridge and arriving there by 20:45, is there anything like that?"
    },
    {
        "System-1": "There are over 1,000 trains like that.  Where will you be departing from?",
        "Dialog_Act-1": {
            "Train-Inform": [["Choice", "over 1"], ["Choice", "000"]],"Train-Request": [["Depart", "?"]]
        },
        "User-2": "I am departing from birmingham new street."},
    {
        "System-2": "Can you confirm your desired travel day?",
        "Dialog_Act-2": {
            "Train-Request": [["Day", "?"]]
        },
        "User-3": "I would like to leave on wednesday"
    }
]

my_pipeline = pipeline(
    task=Tasks.task_oriented_conversation,
    model=model_id
)

history_states = [{}]
utter = {}
for step, item in enumerate(test_case):
    utter.update(item)
    result = my_pipeline({
        'utter': utter,
        'history_states': history_states
    })
    print({k: v for k, v in result[OutputKeys.OUTPUT].items() if v != 'none'})  # 输出对话状态，值='none'的 slot 代表当前未提及，默认不显示

    history_states.extend([result[OutputKeys.OUTPUT], {}])
```

#### 对话样例

| 轮次 | 输入                                                                                                                                                                                                                                                                                                                           | 输出（对话状态）                                                                                                            |
|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| 1    | User utterance: I am looking for a place to stay that has a cheap price range it should be in a type of hotel.                                                                                                                                                                                                                  | hotel-type=hotel, hotel-pricerange=cheap                                                                                    |   
| 1    | System response: Okay, do you have a specific area you want to stay in?        |                                                          |
| 1    | System dialog actions: { 'Hotel-Request': [['Area', '?']] }         |                                                          |
| 2    |  User utterance: I just need to make sure it's cheap. oh, and i need parking.                                                                                              | hotel-type=hotel, hotel-pricerange=cheap, hotel-parking=yes                                                                 |
| 2    | System response: I found 1 cheap hotel for you that includes parking. Do you like me to book it?  |  |
| 2    |  System dialog actions: { 'Booking-Inform': [['none', 'none']], 'Hotel-Inform': [['Price', 'cheap'], ['Choice', '1'], ['Parking', 'none']] }  |  |
| 3 |   User utterance: Yes, please. 6 people 3 nights starting on tuesday.  |hotel-type=hotel, hotel-pricerange=cheap, hotel-parking=yes, hotel-book_people=6, hotel-book_day=tuesday, hotel-book_stay=3|
| 3    | System response: I am sorry but I wasn't able to book that for you for Tuesday. Is there another day you would like to stay or perhaps a shorter stay? |  | 
| 3    | System dialog actions: { "Booking-NoBook": [ [ "Day", "Tuesday" ] ], "Booking-Request": [ [ "Stay", "?" ], [ "Day", "?" ] ] } |  | 
| 4   | User utterance: how about only 2 nights.  |hotel-type=hotel, hotel-pricerange=cheap, hotel-parking=yes, hotel-book_people=6, hotel-book_day=tuesday, hotel-book_stay=2|
| 4    | System response: Booking was successful. Reference number is : 7GAWK763. Anything else I can do for you? |  | 
| 4    | System dialog actions: { "general-reqmore": [ [ "none", "none" ] ], "Booking-Book": [ [ "Ref", "7GAWK763" ] ] } |  | 
| 5   | User utterance: No, that will be all. Good bye.  |hotel-type=hotel, hotel-pricerange=cheap, hotel-parking=yes, hotel-book_people=6, hotel-book_day=tuesday, hotel-book_stay=2|



### 模型局限性以及可能的偏差
由于下游任务是都是限定领域的对话，因此用户需要在该任务领域相关的场景里使用，不宜进行开放域测试。请参考范例更多了解相关场景。

## 训练数据介绍
MultiWOZ2.2 是 MultiWOZ2.0 经过标注label 修正得到的数据，是一个多领域的多轮对话数据，是任务型对话最受欢迎的benchmark，一共包括了1万个对话 sessions。
该数据集中包含了对话状态跟踪任务，专门用于本项目的模型训练。

## 模型训练流程

### 预处理
输入文本将全部转化为小写的形式，然后采用一个预设词表大小为35022的WordPiece子词分词算法进行分词。输入模型的对话将被处理成和 TripPy 模型一样的形式
```text
<CLS> user query <SEP> system response <SEP >dialog history <SEP>
```

### 训练
模型采用8张40G显存的A100进行预训练，总计优化了30个epoch。模型架构采用隐藏层状态维度为768的12层Transformer模块，输入的对话上下文和系统回复语句的最大长度分别限制为256和50。批处理大小设置为128，采用初始学习率设置为1e-5的AdamW优化器进行优化。随机失活概率设置为0.2。在下游任务的微调阶段，我们采用了和 TripPy 一样的设置。


## 数据评估及结果
模型进行对话状态追踪下游任务，在MultiWOZ2.2数据集上取得了最高指标：

### Task：对话状态跟踪 Dialog State Tracking
| Dataset Name | Joint Goal Accuracy |
|:-------------------:|:-----:|
|       MultiWOZ2.2       | 57.50 |

> **NOTE**: 对话状态跟踪是要预测多个键值对是否正确，因此指标是联合准确率 Joint Goal Accuracy，评判每轮所有键值对正确才算对。


### 相关论文以及引用信息

```bib
@article{he2022unified, 
    title={Unified Dialog Model Pre-training for Task-Oriented Dialog Understanding and Generation}, 
    author={He, Wanwei and Dai, Yinpei and Yang, Min and Huang, Fei and Si, Luo and Sun, jian and Li, Yongbin}, 
    journal={SIGIR}, 
    year={2022} 
}
```
