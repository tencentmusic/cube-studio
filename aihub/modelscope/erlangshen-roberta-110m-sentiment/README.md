# Erlangshen-RoBERTa-110M-Sentiment

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

中文的RoBERTa-wwm-ext-base在数个情感分析任务微调后的版本

This is the fine-tuned version of the Chinese RoBERTa-wwm-ext-base model on several sentiment analysis datasets.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | RoBERTa |      110M      |    情感分析 Sentiment     |

## 模型信息 Model Information

基于[chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext)，我们在收集的8个中文领域的情感分析数据集，总计227347个样本上微调了一个Semtiment版本。

Based on [chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext), we fine-tuned a sentiment analysis version on 8 Chinese sentiment analysis datasets, with totaling 227,347 samples.

### 下游效果 Performance

|    模型 Model   | ASAP-SENT    |  ASAP-ASPECT  | ChnSentiCorp    |
| :--------:    | :-----:  | :----:  | :-----:   | 
| Erlangshen-RoBERTa-110M-Sentiment | 97.77     |   97.31    | 96.61     |
| Erlangshen-RoBERTa-330M-Sentiment | 97.9      |   97.51    | 96.66      |  
| Erlangshen-MegatronBERT-1.3B-Sentiment | 98.1     |   97.8    | 97      | 


## 使用 Usage

``` python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(Tasks.text_classification, 'Fengshenbang/Erlangshen-RoBERTa-110M-Sentiment')
p(input='今天心情不好')

```

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2209.02970):

```text
@article{fengshenbang,
  author    = {Junjie Wang and Yuxiang Zhang and Lin Zhang and Ping Yang and Xinyu Gao and Ziwei Wu and Xiaoqun Dong and Junqing He and Jianheng Zhuo and Qi Yang and Yongfeng Huang and Xiayu Li and Yanghan Wu and Junyu Lu and Xinyu Zhu and Weifeng Chen and Ting Han and Kunhao Pan and Rui Wang and Hao Wang and Xiaojun Wu and Zhongshen Zeng and Chongpei Chen and Ruyi Gan and Jiaxing Zhang},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

也可以引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

You can also cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```