# Erlangshen-RoBERTa-110M-Similarity

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

中文的RoBERTa-wwm-ext-base在数个相似度任务微调后的版本

This is the fine-tuned version of the Chinese RoBERTa-wwm-ext-base model on several similarity datasets.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | RoBERTa |      110M      |    相似度 Similarity     |

## 模型信息 Model Information

基于[chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext-base)，我们在收集的20个中文领域的改写数据集，总计2773880个样本上微调了一个Similarity版本。

Based on [chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext-base), we fine-tuned a similarity version on 20 Chinese paraphrase datasets, with totaling 2,773,880 samples.

### 下游效果 Performance

|    Model   | BQ    |  BUSTM  | AFQMC    |
| :--------:    | :-----:  | :----:  | :-----:   | 
| Erlangshen-RoBERTa-110M-Similarity | 85.41     |   95.18    | 81.72     |
| Erlangshen-RoBERTa-330M-Similarity | 86.21      |   99.29    | 93.89      |  
| Erlangshen-MegatronBERT-1.3B-Similarity | 86.31      |   -    | -      |   

## 使用 Usage

``` python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(Tasks.text_classification, 'Fengshenbang/Erlangshen-RoBERTa-110M-Similarity')
p(input='今天心情不好[SEP]今天很开心')

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