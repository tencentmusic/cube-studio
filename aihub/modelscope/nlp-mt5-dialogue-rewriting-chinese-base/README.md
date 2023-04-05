

# 中文多轮对话改写任务说明
多轮对话改写任务主要解决对话中的指代和省略问题，输入对话上下文，输出改写后的问题（示例参考代码范例）；
该模型基于[google/mt5-base](https://huggingface.co/google/mt5-base)在公开+业务数据集上finetune而得，适用于开放域对话场景。

## 模型描述
模型结构与T5模型一致，模型结构的详细介绍，参考：[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)；
基座模型使用[google/mt5-base](https://huggingface.co/google/mt5-base)，并在公开+业务数据集上finetune得到多轮对话改写模型。


## 期望模型使用方式以及适用范围
本模型主要用于输入对话上下文生成改写后的问题，具体调用方式请参考代码示例。

### 如何使用
在安装完成Modelscope之后即可使用多轮对话改写的能力


#### 代码范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipeline_ins = pipeline(task=Tasks.text2text_generation, model='damo/nlp_mt5_dialogue-rewriting_chinese-base',model_revision='v1.0.1')
result = pipeline_ins(input='杨阳胖吗[SEP]我一个同学叫杨阳[SEP]他多少斤')
print (result)
```


## 模型局限性以及可能的偏差
模型在开放域对话改写数据集上进行训练，在其他领域表现有待验证，请谨慎使用；