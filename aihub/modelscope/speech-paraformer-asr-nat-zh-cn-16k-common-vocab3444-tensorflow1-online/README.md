
# Paraformer模型介绍

## 模型描述

[//]: # (Paraformer 模型是一种非自回归（Non-autoregressive）端到端语音识别模型。)

[//]: # (非自回归模型相比于自回归模型，可以对整条句子并行输出目标文字，具有更高的计算效率，尤其采用GPU解码。)

[//]: # (Paraformer模型相比于其他非自回归模型，不仅具有高效的解码效率，在模型参数可比的情况下，模型识别性能与SOTA的自回归模型相当。)

[//]: # (目前Paraformer在如下数据集进行了性能验证：[AISHELL-1]&#40;http://www.openslr.org/33/&#41;、[AISHELL-2]&#40;https://www.aishelltech.com/aishell_2&#41;、[WenetSpeech]&#40;http://www.openslr.org/121/&#41;、阿里内部工业大数据。)

近年来，随着端到端语音识别的流行，基于Transformer结构的语音识别系统逐渐成为了主流。然而，由于Transformer是一种自回归模型，需要逐个生成目标文字，计算复杂度随着目标文字数量线性增加，限制了其在工业生产中的应用。针对Transoformer模型自回归生成文字的低计算效率缺陷，学术界提出了非自回归模型来并行的输出目标文字。根据生成目标文字时，迭代轮数，非自回归模型分为：多轮迭代式与单轮迭代非自回归模型。其中实用的是基于单轮迭代的非自回归模型。对于单轮非自回归模型，现有工作往往聚焦于如何更加准确的预测目标文字个数，如CTC-enhanced采用CTC预测输出文字个数，尽管如此，考虑到现实应用中，语速、口音、静音以及噪声等因素的影响，如何准确的预测目标文字个数以及抽取目标文字对应的声学隐变量仍然是一个比较大的挑战；另外一方面，我们通过对比自回归模型与单轮非自回归模型在工业大数据上的错误类型（如下图所示，AR与vanilla NAR），发现，相比于自回归模型，非自回归模型，在预测目标文字个数方面差距较小，但是替换错误显著的增加，我们认为这是由于单轮非自回归模型中条件独立假设导致的语义信息丢失。于此同时，目前非自回归模型主要停留在学术验证阶段，还没有工业大数据上的相关实验与结论。 

<div align=center>
<img src="fig/error_type.png" width="600" height="400"/>
</div>

因此，为了解决上述问题，我们设计了一种具有高识别率与计算效率的单轮非自回归模型Paraformer。针对第一个问题，我们采用一个预测器（Predictor）来预测文字个数并通过CIF机制来抽取文字对应的声学隐变量。针对第二个问题，受启发于机器翻译领域中的GLM（Glancing language model），我们设计了一个基于GLM的采样器模块来增强decoder模型对上下文语义的建模。除此之外，我们还设计了一种生成负样本策略来引入MWER区分性训练。

其模型结构如下图所示：

<div align=center><img width="500" height="450" src="fig/struct.png"/></div>

其核心点主要有：

Predictor 模块：基于 CIF 的 Predictor 来预测语音中目标文字个数以及抽取目标文字对应的声学特征向量

Sampler：通过采样，将声学特征向量与目标文字向量变换成含有语义信息的特征向量，配合双向的 Decoder 来增强模型对于上下文的建模能力

基于负样本采样的 MWER 训练准则

更详细的描述见：[论文](https://arxiv.org/abs/2206.08317)，[论文解读](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw)

## 使用方式以及适用范围

运行范围
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

使用方式
- 直接推理：可以直接对输入音频进行解码，输出目标文字。
- 微调：加载训练好的模型，采用私有或者开源数据进行模型训练。

使用范围与目标场景
- 该模型为伪流式模型，可以用来评估流式模型效果，配合runtime才可以实现真正的实时识别。

### 如何使用
#### 输入音频格式
输入音频支持wav与pcm格式音频，以wav格式输入为例，支持以下几种输入方式：

- wav文件路径，例如：data/test/audios/asr_example.wav
- wav二进制数据，格式bytes，例如：用户直接从文件里读出bytes数据或者是麦克风录出bytes数据
- wav文件url，例如：https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example.wav
- wav文件测试集，目录结构树必须符合如下要求：
    ```
    datasets directory 
    │
    └───wav
    │   │
    │   └───test
    │       │   xx1.wav
    │       │   xx2.wav
    │       │   ...
    │   
    └───transcript
        │   data.text  # hypothesis text
    ```

#### api调用范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_16k_pipline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab3444-tensorflow1-online')

rec_result = inference_16k_pipline(audio_in='https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/asr_example.wav')
print(rec_result)
```

如果是pcm格式输入音频，调用api时需要传入音频采样率参数audio_fs，例如：
```python
rec_result = inference_16k_pipline(audio_in='https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/asr_example.pcm', audio_fs=16000)
```


### 模型局限性以及可能的偏差

考虑到特征提取流程和工具以及训练工具差异，会对CER的数据带来一定的差异（<0.1%），推理GPU环境差异导致的RTF数值差异。

## 训练数据介绍

5万小时16K通用数据

## 模型训练流程

在AISHELL-1与AISHELL-2等学术数据集中，采用随机初始化的方式直接训练模型。
在工业大数据上，建议加载预训练好的自回归端到端模型作为初始，训练Paraformer。

### 预处理

可以直接采用原始音频作为输入进行训练，也可以先对音频进行预处理，提取FBank特征，再进行模型训练，加快训练速度。

## 数据评估及结果

|    model     | clean（CER%） | common(CER%) | RTF   |
|:------------:|:---------:|:---------:|:------:|
| Paraformer   |   9.73    |   12.96   | 0.0093 |


### 相关论文以及引用信息

```BibTeX
@inproceedings{gao2022paraformer,
  title={Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition},
  author={Gao, Zhifu and Zhang, Shiliang and McLoughlin, Ian and Yan, Zhijie},
  booktitle={INTERSPEECH},
  year={2022}
}
```
