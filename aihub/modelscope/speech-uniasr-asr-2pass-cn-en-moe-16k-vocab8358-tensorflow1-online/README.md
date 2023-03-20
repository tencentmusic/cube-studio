
# UniASR 模型介绍

## 模型描述

中英自由说模型旨在让一套语音识别系统能高精度识别中文、英文、中英文混合语音。我们在UniASR模型基础上，通过借鉴混合专家系统(Mixture of Experts, MOE)的思想，在语音识别系统里面内嵌中、英文专家模块，使得语音识别系统无需切换就能支持中文、英文和中英混读的语音识别。UniASR 模型是一种2遍刷新模型（Two pass）端到端语音识别模型。日益丰富的业务需求，不仅要求识别效果精度高，而且要求能够实时地进行语音识别。一方面，离线语音识别系统具有较高的识别准确率，但其无法实时的返回解码文字结果，并且，在处理长语音时，容易发生解码重复的问题，以及高并发解码超时的问题等；另一方面，流式系统能够低延时的实时进行语音识别，但由于缺少下文信息，流式语音识别系统的准确率不如离线系统，在流式业务场景中，为了更好的折中实时性与准确率，往往采用多个不同时延的模型系统。为了满足差异化业务场景对计算复杂度、实时性和准确率的要求，常用的做法是维护多种语音识别系统，例如，CTC系统、E2E离线系统、SCAMA流式系统等。在不同的业务场景使用不同的模型和系统，不仅会增加模型生产成本和迭代周期，而且会增加引擎以及服务部署的维护成本。因此，我们设计了离线流式一体化语音识别系统——UniASR。UniASR同时具有高精度和低延时的特点，不仅能够实时输出语音识别结果，而且能够在说话句尾用高精度的解码结果修正输出，与此同时，UniASR采用动态延时训练的方式，替代了之前维护多套延时流式系统的做法。通过设计UniASR语音识别系统，我们将之前多套语音识别系统架构统一为一套系统架构，一个模型满足所有业务场景，显著的降低了模型生产和维护成本。
其模型结构如下图所示：


<div align=center>
<img src="fig/UniASR.png" width="850" height="270"/>
</div>

UniASR模型结构如上图所示，包含离线语音识别部分和流式语音识别部分。其中，离线与流式部分通过共享一个动态编码器（Encoder）结构来降低计算量。流式语音识别部分是由动态时延 Encoder 与流式解码器（Decoder）构成。动态时延 Encoder 采用时延受限有句记忆单元的自注意力（LC-SAN-M）结构；流式 Decoder 采用动态 SCAMA 结构。离线语音识别部分包含了降采样层（Sride Conv）、Big-Chunk Encoder、文本Encoder与SCAMA Decoder。为了降低刷新输出结果的尾点延时，离线识别部分采用大Chunk 流式结构。其中，Stride Conv结构是为了降低计算量。文本 Encoder 增加了离线识别的语义信息。为了让模型能够具有不同延时下进行语音识别的能力，我们创新性地设计了动态时延训练机制，使得模型能够同时满足不同业务场景对延时和准确率的要求。
根据业务场景特征，我们将语音识别需求大致分为3类： 

    低延迟实时听写：如电话客服，IOT语音交互等，该场景对于尾点延迟非常敏感，通常需要用户说完以后立马可以得到识别结果。  
    流式实时听写：如会议实时字幕，语音输入法等，该场景不仅要求能够实时返回语音识别结果，以便实时显示到屏幕上，而且还需要能够在说话句尾用高精度识别结果刷新输出。  
    离线文件转写：如音频转写，视频字幕生成等，该场景不对实时性有要求，要求在高识别准确率情况下，尽可能快的转录文字。  

为了同时满足上面3种业务场景需求，我们将模型分成3种解码模式，分别对应为：  

    fast 模式：只有一遍解码，采用低延时实时出字模式；  
    normal 模式：2遍解码，第一遍低延时实时出字上屏，第二遍间隔3～6s（可配置）对解码结果进行刷新；  
    offline 模式：只有一遍解码，采用高精度离线模式；  

在模型部署阶段，通过发包指定该次语音识别服务的场景模式和延时配置。这样，通过UniASR系统，我们统一了离线流式语音识别系统架构，提高模型识别效果的同时，不仅降低了模型生产成本和迭代周期，还降低了引擎以及服务部署维护成本。目前我们提供的语音识别服务基本都是基于UniASR。  

## 使用方式以及适用范围

运行范围
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

使用方式
- 直接推理：可以直接对输入音频进行解码，输出目标文字。
- 微调：加载训练好的模型，采用私有或者开源数据进行模型训练。

使用范围与目标场景
- 建议输入语音时长在20s以下。

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
    model='damo/speech_UniASR_asr_2pass-cn-en-moe-16k-vocab8358-tensorflow1-online')

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

5万小时16K中文通用数据、5万小时16K英文通用数据

## 模型训练流程

在AISHELL-1与AISHELL-2等学术数据集中，采用随机初始化的方式直接训练模型。
在工业大数据上，建议加载预训练好的离线端到端模型作为初始，训练UniASR。

### 预处理

可以直接采用原始音频作为输入进行训练，也可以先对音频进行预处理，提取FBank特征，再进行模型训练，加快训练速度。

## 数据评估及结果

|        model        | test_cn（CER%）| test_en (CER%) | test_cs (CER%) |
|:-------------------:|:---------:|:---------:|:---------:|
| offline |   10.01   |   10.31   |   12.04   |
| normal  |   10.73   |   10.71   |   12.48   |
| fast(900ms)   |   12.89   |   11.53   |   13.31   |

### 相关论文以及引用信息

```BibTeX
@inproceedings{gao2020universal,
  title={Universal ASR: Unifying Streaming and Non-Streaming ASR Using a Single Encoder-Decoder Model},
  author={Gao, Zhifu and Zhang, Shiliang and Lei, Ming and McLoughlin, Ian},
  booktitle={arXiv preprint arXiv:2010.14099},
  year={2010}
}
```
