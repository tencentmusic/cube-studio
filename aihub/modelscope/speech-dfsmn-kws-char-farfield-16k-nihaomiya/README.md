
# FSMN远场唤醒模型介绍

## 问题背景
关键词检测（keyword spotting, KWS），即我们通常所说的语音唤醒，指的是一系列从实时音频流中检测出若干预定义关键词的技术。随着远讲免提语音交互（distant-talking hands free speech interaction）技术的发展，关键词检测及其配套技术也变得越来越重要。类比于人和人交互时先喊对方的名字一样，关键词就好比智能设备的"名字"，而关键词检测模块则相当于交互流程的触发开关。

针对各类AIoT应用来说，由于需要在硬件条件有限的设备端对音频流进行实时监听，所以关键词检测模块必须做到"低资源、高性能"。所谓低资源，指的是全套算法所需的算力、功耗、存储、网络带宽等资源应当做到尽量节省，以满足实际设备硬件条件的限制；而所谓高性能，则是要求智能设备在包含各种设备回声、人声干扰、环境噪声、房间混响的实际应用场景中也能具有较高的唤醒率和较低的虚警率，同时具有较小的唤醒事件响应延迟。针对实际场景中各种不利声学因素的影响，只靠关键词检测本身是无法应对的，所以关键词检测技术一般需要配合语音增强（speech enhancement）技术来使用，并且语音增强和关键词检测还需要实现匹配训练和联合优化才能发挥出更好的性能表现。

本项目以你好米雅等开源数据集为基础，提供了一种基于盲源分离（blind source separation, BSS）理论框架的语音增强方法，以及一种可扩展的多通道关键词检测与通道选择模型。同时还提供了配套的数据模拟，模型训练以及测试工具链。该模型参数量为123k，适合于低资源嵌入式应用。并且该模型还实现了和语音增强算法的匹配训练，进一步提升了模型在实际系统中的性能。

本项目的总体框架如图1所示，其中主要分为语音增强和关键词检测两个部分。首先，语音增强模块以多通道麦克风信号<img height="22" src="figs/fx.jpg"/>和单路参考信号r为输入，经过去混响、回声消除、盲源分离、增益控制等操作后，输出多路分离后的信号<img height="22" src="figs/fy.jpg"/>。此时我们可以认为其中一路输出是包含"你好米雅"关键词信噪比较高的目标语音，而其它输出则是信噪比较低的噪声或干扰信号。之后多路信号经过关键词检测模块进行处理，检测其中的关键词音频，抛出唤醒事件，并选择关键词信噪比最高的通道，即n'，供后续交互流程使用。

<div align=center>
<img width="780" src="figs/system_overview.png"/>
</div>
<center>图1 系统框架图。</center>


## 模型描述

### 语音增强
在语音增强系统中，各个算法子模块通常采用级联的方式进行组合。级联架构的优点在于其中使用了"分而治之"的思想，将原始的复杂问题拆解为若干个稍简单的子问题，并采用相应的算法进行处理。各个算法子模块独立运作，每个子模块只需处理好自己的任务即可，无须关心其它子模块的工作原理，子模块之间除了输入输出数据之外也不存在其它的信息传递。所以采用级联架构能简化整个系统的设计难度，并且增加了替换兼容接口的算法子模块的灵活性。

但是，级联架构的系统也存在一些缺点：由于不同的算法其目标函数和优化方法也各不相同，并且每个算法子模块都是独立运作的，所以系统中的算法子模块各自收敛到其目标函数的最优解后，并不能代表整体系统性能也达到了最优。针对上述问题，本项目使用了基于盲源分离统一框架的语音增强算法，将去混响（dereverberation, DR）、回声消除（acoustic echo cancellation, AEC）、以及声源的分离问题都统一到了盲源分离的理论框架中，从而实现了目标函数和优化方法的统一，达到联合优化的目的。该算法的信号模型示意图如图2所示，其中A表示混合矩阵，B表示分离矩阵。读者可以查看参考文献1以了解更多相关信息。

<div align=center>
<img width="780" src="figs/signal_model.png"/>
</div>
<center>图2 盲源分离统一框架信号模型示意图。</center>

### 关键词检测
本项目提供的关键词检测算法框架主要由三个部分组成：特征提取、声学模型（acoustic model, AM）、解码器（decoder）。特征提取用于从增强过后的语音信号中提取声学特征；声学模型一般由深度神经网络构成，以特征为输入，并预测各个关键词建模单元的观测概率。本项目中采用基于FSMN（feedforward sequential memory network）的网络结构，如图3所示；解码器则用于对建模单元观测概率进行平滑处理，并从概率曲线的变化逻辑中找出发音单元的变化序列，并以此作为某个关键词出现的标志。

<div align=center>
<img width="780" src="figs/am.png"/>
</div>
<center>图3 声学模型结构示意图。</center>

本项目中采用汉字建模，关键词"你好米雅"为四个建模单元，分别对应图3中的A、B、C、D，而Filler则建模了非关键词音频。图3中的模型架构还可以通过max pooling操作中获得关键词信号质量最好的通道序号n'，该通道的数据将用于后续的语音交互流程中。

### 数据模拟
本项目中的多通道模型需要使用多通道数据进行训练。由于真实的多通道数据较难获得，所以需要采用数据模拟的方式来生成海量多通道训练数据。同时，为了使得语音增强与关键词检测能够实现匹配训练，我们还需要将短句级别的音源通过随机拼接的方式扩展为分钟级别的长音频，从而保证语音增强算法的收敛性。数据模拟的框架图如图4所示。

<div align=center>
<img width="780" src="figs/simu.png"/>
</div>
<center>图4 数据模拟系统框架图。</center>



## 期望模型使用方式以及适用范围

您可以在模型介绍页面右侧快速体验模型效果，也可以参考下面步骤自己编码调用模型。

注意：模型本身只支持多通道音频，但为了让用户亲自录音体验唤醒效果，我们给推理流程加了兼容处理。这个兼容效果无法达到真实场景的唤醒率水平，而且**建议在录制测试音频时，开始录制后等待3秒左右再喊唤醒词，最后一次唤醒词说完后再等待3秒左右才停止录音。**

### 如何使用

在安装ModelScope完成之后即可使用```speech_dfsmn_kws_char_farfield_16k_nihaomiya```进行推理。

模型输入为3通道音频，其中前2个通道是双麦的麦克风阵列录制的音频，第3个通道是设备同时播放的音频作为参考信号。音频格式为16000Hz采样率，16位宽，PCM编码的wav文件。

#### 环境准备

* 本模型支持Linux，Windows平台下的推理和训练，MacOS平台支持正在开发中。
* 本模型使用了三方库SoundFile进行wav文件处理，**在Linux系统上用户需要手动安装SoundFile的底层依赖库libsndfile**，在Windows和MacOS上会自动安装不需要用户操作。详细信息可参考[SoundFile官网](https://github.com/bastibe/python-soundfile#installation)。以Ubuntu系统为例，用户需要执行如下命令:

```shell
sudo apt-get update
sudo apt-get install libsndfile1
```

#### 代码范例

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


kws = pipeline(
    Tasks.keyword_spotting,
    model='damo/speech_dfsmn_kws_char_farfield_16k_nihaomiya')
# you can also use local file path
result = kws('https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/3ch_nihaomiya10.wav')
print(result)
```

### 模型局限性以及可能的偏差
本项目只使用了开源的正负样本以及噪声数据进行训练，所以模型性能受数据所限。在实际应用中应该进一步扩充数据来源并采集场景噪声进行训练。

## 训练数据介绍
本项目中所使用的数据均为开源数据，包括：
- HI-MIA (http://www.openslr.org/85/) ：你好米雅唤醒词数据
- AISHELL2 (https://www.aishelltech.com/aishell_2) ：非关键词语音负样本数据
- MUSAN (https://www.openslr.org/17/) ：噪声数据
- DNS-Challenge (https://github.com/microsoft/DNS-Challenge) ：噪声数据

本项目所使用的多通道长音频测试集同样使用上述开源数据通过模拟的方法来构造。其中，正样本测试集包含6小时音频，共包含2090个关键词，信干比（interference ratio, SIR）、信回比（signal-to-echo ratio, SER）、信噪比（signal-to-noise ratio, SNR）区间分别为：[-15, 5]、[-25, 5]、[-8, 15] dB；负样本测试集包含100小时的音频，模拟环境与正样本测试集相同，但不包含关键词。测试集与训练集中的正负样本音源没有重叠。

## 模型训练流程

我们封装了ModelScope的模型训练能力，再增加数据处理，效果评测，流程控制等辅助功能，连同一些相关工具打包成[唤醒模型训练套件](https://github.com/alibaba-damo-academy/kws-training-suite)，已经在Github上开源，欢迎有兴趣的开发者试用。

为了达到更好的唤醒效果，训练套件默认会做两轮训练和评测。第一轮训出的模型根据评测结果选出最优的模型作为基础，第二轮再继续finetune。大致流程如下图所示，更多详细信息请参考套件说明文档。

<div align=center>
<img width="400" src="figs/training_pipeline.jpg"/>
</div>
<center>图5 模型训练流程图</center>


## 数据评估及结果

当前模型在通道数为1到4（N = 1, ..., 4）的模拟数据集上的性能表现如图5中的ROC（receiver operating characteristic）曲线所示，其中N = 1时无法使用盲源分离算法对混合信号进行分离。

![proposed_comp.png](figs/proposed_comp.png)
<center>图6 当前模型的性能。</center>

### 相关论文以及引用信息

```bib
@inproceedings{na2022joint,
  title={Joint Ego-Noise Suppression and Keyword Spotting on Sweeping Robots},
  author={Na, Yueyue and Wang, Ziteng and Wang, Liang and Fu, Qiang},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7547--7551},
  year={2022},
  organization={IEEE},
  url = {https://github.com/nay0648/ego2022}
}
```

```bib
@inproceedings{na2021joint,
  title={Joint Online Multichannel Acoustic Echo Cancellation, Speech Dereverberation and Source Separation},
  author={Na, Yueyue and Wang, Ziteng and Liu, Zhang and Tian, Biao and Fu, Qiang},
  booktitle={Interspeech},
  year={2021},
  url = {https://github.com/nay0648/unified2021}
}
```
