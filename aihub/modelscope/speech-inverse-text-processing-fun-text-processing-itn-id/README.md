

# 印尼语逆文本正则化模型

## 模型描述

印尼语逆文本正则化模型是基于[FunTextProcessing](https://github.com/alibaba-damo-academy/FunASR/tree/main/fun_text_processing) 开源代码库生成，用于印尼语语音识别模型结果后处理中的逆文本正则化部分。


## 多语言逆文本正则化&文本正则化

逆文本正则化（Inverse Text Normalization）和文本正则化（Text Normalization）是语音交互系统中必不可少的部分。逆文本正则化（ITN）广泛应用于语音识别结果的文本后处理模块，实现从口语域到书面域的文字的转换，使显示的文字更加符合人的阅读习惯。文本正则化（TN）广泛用于语音合成系统的前端数据处理。

当前被学术界和工业界广泛使用的逆文本正则化（ITN）和文本正则化（TN）系统有以下三大类。
1. 基于语法规则WFST的系统 这种系统由大量特定语言的语法组成，该方法的优点是准确性可控，可以快速修复语法中的Badcase,该方法的不足在于没有特定上下文情况下，对于产生歧义的文本不够鲁棒。
2. 基于神经网络模型的系统 这种系统需要大量的训练数据，需要这种语言的语法规则，将书面域转换为相应的口语域的数据，以产生大量的覆盖范围更广的数据。同时其方法的另外一个主要的缺点是无法修复转变的错误，修复Badcase的方法不如基于语法规则的方式来的简单。
3. 混合使用语法和神经网络系统 这种混合系统中当系统没有找到匹配的语法规则时，才利用神经网络模型进行转换。该方法比较好的权衡了规则和模型的优劣，但对系统的计算资源等提出更高的要求。

多语言逆文本正则化和文本正则化系统示意图如下：

<div align=center>
<img src="fig/struct.png" width="821" height="327"/>
</div>

综合上述分析，我们选择利用基于语法的WFST方案，在ModelScope平台上开源包括中、英、日、韩、印尼等十多种不同语言的ITN规则。对于每一种语言的ITN&TN，由两大部分组成，一部分是Tagger, 另一部分是Verbalizer. Tagger的作用就相当于分类器，对于输入的文本利用不同类型的规则去Tag获得输入文本的类型。在Tag的过程中，主要根据WFST获得最短的路径。不同类型规则的权值是也会影响到Tag获得的输入文本的Tag类型。下图给出了利用WFST输入是"twenty three"情况下，利用最短路径获得输出“23”，而不是输出“20 3”的例子。

<div align=center>
<img src="fig/fst.png" width="1282" height="200"/>
</div>

Verbalizer则是将parse和调序之后的tokens调用相关类型的Verbalizer输出，获得最后的结果。同时基于[FunTextProcessing](https://github.com/alibaba-damo-academy/FunASR/tree/main/fun_text_processing)，开源了设计和生成这些ITN规则的工具。该工具提供了安装、测试、导出的python工具包。具体的使用方法可以参考FunTextProcessing中的[README.md](https://github.com/alibaba-damo-academy/FunASR/blob/main/fun_text_processing/README.md)。

## 期望模型使用方式以及适用范围

### 运行范围
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

### 使用方式
- 直接推理：可以直接输入文字，输出逆正则化转换之后的文字，例如：输入数字（zero - nine）转换为阿拉伯数字（0-9）输出。
- 修改：利用[FunTextProcessing](https://github.com/alibaba-damo-academy/FunASR/tree/main/fun_text_processing)相应语言的不同语法规则进行修改。

### 使用范围与目标场景
- 相应语言语音识别结果的后处理。

### 如何使用
#### api调用范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

itn_inference_pipline = pipeline(
    task=Tasks.inverse_text_processing,
    model='damo/speech_inverse_text_processing_fun-text-processing-itn-id')

itn_result = itn_inference_pipline(text_in='seratus dua puluh tiga')
print(itn_result)
```

### 模型局限性以及可能的偏差
由于此模型由基于该语种的ITN语法规则编译而成，可能会受限于现有语法规则覆盖不全，或多个语法规则相互冲突的情况。考虑到此局限，我们开源了ITN代码库，欢迎大家在我们的[FunTextProcessing](https://github.com/alibaba-damo-academy/FunASR/tree/main/fun_text_processing)开源代码贡献更多语法规则或加入更多语言ITN规则。

## 训练数据介绍

此模型由基于该语种的ITN语法规则编译生成，无需训练数据。

### 预处理
输入文本数据分两类：
- 可直接将语音识别模型输出的文本结果输入ITN模型，输出经过逆文本正则化后的文本。
- 语音识别模型输出文本结果后，可先通过标点模型，加入标点符号，然后将加入标点后的语音识别文本结果输入ITN模型。

### 后处理
逆文本正则化输出结果后续可再做顺滑或纠错处理。

## 数据评估及结果

### 相关论文以及引用信息

```github
https://github.com/alibaba-damo-academy/FunASR/tree/main/fun_text_processing
```
