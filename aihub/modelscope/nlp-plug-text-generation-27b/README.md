
# 大规模中文理解和生成联合模型PLUG

PLUG (Pre-training for Language Understanding and Generation) 是一个270亿参数的大规模中文理解和生成联合预训练模型。

**Demo体验，请点击右侧进入AI写手创空间!!!**

## 模型描述

PLUG是有海量高质量中文文本预训练得到的理解和生成联合模型。PLUG的训练由两阶段组成。首先我们训练了一个24层的基于[StructBERT](https://arxiv.org/abs/1908.04577)的encoder，然后我们基于此训练了一个24+6层的[PALM](https://arxiv.org/pdf/2004.07159.pdf?fbclid=IwAR0BNl1IzR5bhcuEbyfNw2UN7MApHFoFP3BN40FKkW8x3bqolK_HilU293I) encoder-decoder。这使得模型既可以通过finetune来处理文本分类、序列标注等自然语言理解（NLU）任务，也可以用来处理自然语言生成（NLG）的任务。

![PLUG architecture](resources/architecture.png)

## 期望模型使用方式以及适用范围
本模型可直接用于文本生成，也可以通过finetune用于各类文本理解的任务。用户可以自行尝试各种输入文档。具体调用方式请参考代码示例。

### 如何使用
在安装完成ModelScope-lib之后即可使用PLUG的能力。

#### 依赖安装
我们将PLUG模型依赖的Megatron相关代码打包到了单独的包中，可以通过以下命令进行安装：
```shell
pip install megatron_util -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

#### 代码范例
此范例为单机8卡(GPU)示例，运行时每张GPU约占用显存12G。

1. 通过model_id获取默认model_dir
```python
from modelscope.hub.snapshot_download import snapshot_download
model_id = 'damo/nlp_plug_text-generation_27B'
model_dir = snapshot_download(model_id)
print(model_dir)
```
2. 将模型二进制文件下载至model_dir/model，下载地址获取：https://github.com/alibaba/AliceMind/tree/main/PLUG#pre-trained-model-download

3. 模型调用
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    input = '段誉轻挥折扇，摇了摇头，说道：“你师父是你的师父，你师父可不是我的师父。"'
    model_id = 'damo/nlp_plug_text-generation_27B'
    pipe = pipeline(Tasks.text_generation, model=model_id)
    pipe.models = []

    # out_length为期望的生成长度，最大为512
    result = pipe(input, out_length=256)
    print(result)
```

### 模型局限性以及可能的偏差
模型训练数据有限，效果可能存在一定偏差。

## 训练数据介绍
数据来源于[https://huggingface.co/datasets/wikipedia](https://huggingface.co/datasets/wikipedia)和[https://commoncrawl.org/](https://commoncrawl.org/)

## 模型训练流程
在中文wiki/ Common crawl等无监督数据上，通过"模型描述"章节介绍的训练任务训练了约300B字得到。
### 预处理
暂无
### 训练
下面是基于PLUG-27B模型在dureader-robust生成数据集上二次开发训练

#### 训练环境准备
PLUG finetune对机器有较高要求。此范例为单机8卡(GPU)示例，运行时每张GPU约占用显存近32G。

运行环境您可以使用modelscope提供的[基础镜像](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)。此外，我们将PLUG模型依赖的Megatron相关代码打包到了单独的包中，可以通过以下命令进行安装：
```shell
pip install megatron_util -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
我们还使用了deepspeed：
```shell
pip install deepspeed==0.7.2
```

#### 代码范例
1. 通过model_id获取默认model_dir
```python
from modelscope.hub.snapshot_download import snapshot_download
model_id = 'damo/nlp_plug_text-generation_27B'
model_dir = snapshot_download(model_id)
print(model_dir)
```
2. 将模型二进制文件下载至model_dir/model，下载地址获取：https://github.com/alibaba/AliceMind/tree/main/PLUG#pre-trained-model-download

3. 以下代码为多卡训练，无法在notebook等环境中直接运行。 需要写成python文件如finetune_plug.py, 运行时需要使用deepspeed命令deepspeed --num_gpus=8 --num_nodes=1 finetune_plug.py整体运行。
详细使用解释可查看文档[大模型使用介绍：大模型finetune最佳实践](https://pre.modelscope.cn/docs/NLG%E5%A4%A7%E6%A8%A1%E5%9E%8B%E4%BD%BF%E7%94%A8%E4%BB%8B%E7%BB%8D)

```python
import os
import tempfile

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

def main():
    # 准备数据集
    from datasets import load_dataset
    dataset_dict = load_dataset('luozhouyang/dureader', 'robust')

    def concat_answer_context(dataset):
        dataset['src_txt'] = dataset['answers']['text'][0] + '[SEP]' + dataset[
            'context']
        return dataset

    train_dataset = dataset_dict['train'].map(concat_answer_context)
    eval_dataset = dataset_dict['validation'].map(concat_answer_context)

    train_dataset = train_dataset \
        .rename_columns({'question': 'tgt_txt'}).remove_columns('context') \
        .remove_columns('id').remove_columns('answers')
    eval_dataset = eval_dataset \
        .rename_columns({'question': 'tgt_txt'}).remove_columns('context') \
        .remove_columns('id').remove_columns('answers')

    # 准备work目录，用以存放log和finetune后的checkpoint文件
    tmp_dir = "plug_work_dir/rank" + os.environ['RANK']
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    model_id = 'damo/nlp_plug_text-generation_27B'

    # 使用plug_trainer进行训练
    kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=tmp_dir)

    trainer = build_trainer(
        name=Trainers.nlp_plug_trainer, default_args=kwargs)
    trainer.train()

if __name__ == '__main__':
    main()
```

### 数据评估及结果

#### Finetune
* [CLUE classification benchmark](https://www.cluebenchmarks.com/classification.html), 结果来自2021/04/20

![clue](resources/clue.png)

* 在问题生成任务上的finetune结果

|Model|Metric|[KBQG](https://github.com/nanduan/NLPCC-KBQA)|[DuReaderQG](https://arxiv.org/abs/1711.05073)|[DuReader-Robust](https://arxiv.org/abs/2004.11142)|
|-----|-----|-------|--------|----|
|plug.zh|BLEU-4|66.30|49.20|42.83|


#### Zero-shot示例
* 小说生成

![novel generation](resources/zero-shot1.png)

* 技术文档撰写

![Scientific Literature generation](resources/zero-shot2.png)

* 常识问答

![common sense q&a](resources/zero-shot3.png)

* Zero-shot分类

![zero-shot classification](resources/zero-shot4.png)

### 开源信息
PLUG同时开源到了[AliceMind](https://github.com/alibaba/AliceMind)，如果我们的工作对您有帮助，欢迎给我们Star。