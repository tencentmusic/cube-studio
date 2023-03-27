
# MossFormer语音分离模型介绍

我们日常可能会遇到在嘈杂环境中进行语言交流的场景，比如在人多的餐厅里或者拥挤的人群中，同时存在着许多不同的说话人的声音，这时听者可能只对一个主说话人的声音感兴趣，其他说话人的声音形成了一种听觉干扰，幸运的是，我们人类的听觉系统具有选择性聆听的功能，可以在多个说话人的声音中选择性的聆听自己感兴趣的声音。1953年，Colin Cherry提出了著名的”鸡尾酒会”问题，即在鸡尾酒会那样的嘈杂环境中，我们似乎也能毫不费力地在其他人的说话声和环境噪声的包围中听到一个人的说话内容，也就是说人类听觉系统能轻易地将一个人的声音和另一个人的分离开来。虽然人类能轻易地分离人声，但事实证明，在这项基本任务中，构建一个能够媲美人类听觉系统的自动化系统是很有挑战性的。“语音分离”（Speech Separation）便来源于“鸡尾酒会问题”，由于麦克风采集的音频信号中除了主说话人之外，还可能包括噪声、其他人说话的声音、混响等干扰。语音分离的目标即是把目标语音从背景干扰中分离出来。其应用范围不仅包括听力假体、移动通信、鲁棒的自动语音以及说话人识别等，最近也被广泛应用在各个语音方向的机器学习场景中。

根据干扰的不同，语音分离任务可以是单纯的多说话人分离，也可以包括噪声消除和解混响等附加任务。在没有噪声和混响的情况下，单纯的语音分离任务已经被研究了几十年，从最初的传统信号处理算法到最近的深度学习算法，算法的分离性能有了明显的进步。我们的研究目标是在现有深度学习算法的基础上，通过融入更先进的门控注意力机制和带记忆的深度卷积网络，从而更有效地对长语音序列进行建模和学习，大幅提升深度学习分离算法的性能。

## 模型描述

MossFormer语音分离模型是基于带卷积增强联合注意力(convolution-augmented joint self-attentions）的门控单头自注意力机制的架构（gated single-head transformer architecture ）开发出来的，通过采用联合局部和全局自注意力架构，同时对局部段执行完整自注意力运算和对全局执行线性化低成本自注意力运算，有效提升当前双路径（Dual-path） Transformer 模型在远程元素交互和局部特征模式的建模能力。除了强大的远程建模能力外，我们还通过使用卷积运算来增强 MossFormer 在局部位置模式的建模能力。因此，MossFormer 的性能明显优于之前的模型，并在 WSJ0-2/3mix 和 WHAM!/WHAMR! 上取得了最先进的结果。

<div align=center>
<img width="640" src="https://modelscope.cn/api/v1/models/damo/speech_mossformer_separation_temporal_8k/repo?Revision=master&FilePath=description/model.jpg&View=true"/>
</div>
<center>图1 MossFormer模型架构图</center>

MossFormer由一个卷积编码器-解码器结构和一个掩蔽网络组成。编码器-解码器结构负责特征提取和波形重建，编码器负责特征提取，由一维 (1D) 卷积层 (Conv1D) 和整流线性单元 (ReLU) 组成，后者将编码输出限制为非负值。解码器是一维转置卷积层，它使用与编码器相同的内核大小和步幅。 掩码网络执行从编码器输出到𝐶组掩码的非线性映射，掩码网络的主组成部分是MossFormer模块，一个MossFormer 模块由四个卷积模块、缩放和偏移操作、联合局部和全局 单头自注意力（SHSA） 以及三个门控操作组成，负责进行长序列的处理。

理论上，我们的模型框架可以支持任意多说话人和任意环境下的语音分离任务，我们在ModelScope上首先开放的是基于两个说话人的纯语音分离模型，其目的是让用户可以在较简单的分离任务上，更快速的搭建和测试我们的模型平台。

## 模型的使用方式

模型pipeline 输入为个8000Hz采样率的单声道wav文件，内容是两个人混杂在一起的说话声，输出结果是分离开的两个单声道音频。

#### 环境准备

* 本模型支持Linux，Windows和MacOS平台。
* 本模型依赖开源库SpeechBrain，由于它对PyTorch版本依赖比较苛刻，因此没有加入ModelScope的默认依赖中，需要用户手动安装：

```shell
# 如果您的PyTorch版本>=1.10 安装最新版即可
pip install speechbrain
# 如果您的PyTorch版本 <1.10 且 >=1.7，可以指定如下版本安装
pip install speechbrain==0.5.12
```

* 本模型还使用了三方库SoundFile进行wav文件处理，**在Linux系统上用户需要手动安装SoundFile的底层依赖库libsndfile**，在Windows和MacOS上会自动安装不需要用户操作。详细信息可参考[SoundFile官网](https://github.com/bastibe/python-soundfile#installation)。以Ubuntu系统为例，用户需要执行如下命令:

```shell
sudo apt-get update
sudo apt-get install libsndfile1
```

#### 代码范例

```python
import numpy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# input可以是url也可以是本地文件路径
input = 'https://modelscope.cn/api/v1/models/damo/speech_mossformer_separation_temporal_8k/repo?Revision=master&FilePath=examples/mix_speech1.wav'
separation = pipeline(
   Tasks.speech_separation,
   model='damo/speech_mossformer_separation_temporal_8k')
result = separation(input)
for i, signal in enumerate(result['output_pcm_list']):
    save_file = f'output_spk{i}.wav'
    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)
```

#### 模型局限性

本模型仅在纯语音数据集上进行训练完成的，因而无法保障在带混响和噪声情况下的分离效果，我们后续会进一步发布可以同时处理带混响和噪声的语音分离模型，敬请期待！

## 训练数据介绍
魔搭社区上开放的模型使用约30小时2人混合语音作为训练数据。混合语音是基于WSJ0数据集生成的，由于WSJ0的License问题无法在这里分享。我们在ModelScope上提供了基于LibriSpeech数据集生成的混合音频，以便您快速开始训练。其中训练集包含约42小时语音，共13900条，大小约7G。

## 模型训练
### 环境准备
ModelScope网站官方提供的Notebook环境已经安装好了所有依赖，能够直接开始训练。如果您要在自己的设备上训练，可以参考上一节的环境准备步骤。环境准备完成后建议运行推理示例代码，验证模型可以正常工作。

### 训练步骤
以下列出的为训练示例代码，其中work_dir可以替换成您需要的路径。训练log会保存在work_dir/log.txt，训练中的模型参数等数据会保存在work_dir/save/CKPT+timestamp路径下。数据训练一遍为一个epoch，默认共训练120个epoch，在硬件配置为20核 CPU 和V100 GPU 的机器上需要约10天。

```python
import os

from datasets import load_dataset

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors.audio import AudioBrainPreprocessor
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

work_dir = './train_dir'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

train_dataset = MsDataset.load(
        'Libri2Mix_8k', split='train').to_torch_dataset(preprocessors=[
        AudioBrainPreprocessor(takes='mix_wav:FILE', provides='mix_sig'),
        AudioBrainPreprocessor(takes='s1_wav:FILE', provides='s1_sig'),
        AudioBrainPreprocessor(takes='s2_wav:FILE', provides='s2_sig')
    ],
    to_tensor=False)
eval_dataset = MsDataset.load(
        'Libri2Mix_8k', split='validation').to_torch_dataset(preprocessors=[
        AudioBrainPreprocessor(takes='mix_wav:FILE', provides='mix_sig'),
        AudioBrainPreprocessor(takes='s1_wav:FILE', provides='s1_sig'),
        AudioBrainPreprocessor(takes='s2_wav:FILE', provides='s2_sig')
    ],
    to_tensor=False)
kwargs = dict(
    model='damo/speech_mossformer_separation_temporal_8k',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=work_dir)
trainer = build_trainer(
    Trainers.speech_separation, default_args=kwargs)
trainer.train()
```

## 数据评估及结果

以下列出的为模型评估代码，其中work_dir是工作路径，要评估的模型必须放在work_dir/save/CKPT+timestamp路径下，程序会搜索路径下的最佳模型并自动加载。

```python
import os

from datasets import load_dataset

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors.audio import AudioBrainPreprocessor
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

work_dir = './train_dir'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

train_dataset = None
eval_dataset = MsDataset.load(
        'Libri2Mix_8k', split='test').to_torch_dataset(preprocessors=[
        AudioBrainPreprocessor(takes='mix_wav:FILE', provides='mix_sig'),
        AudioBrainPreprocessor(takes='s1_wav:FILE', provides='s1_sig'),
        AudioBrainPreprocessor(takes='s2_wav:FILE', provides='s2_sig')
    ],
    to_tensor=False)
kwargs = dict(
    model='damo/speech_mossformer_separation_temporal_8k',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=work_dir)
trainer = build_trainer(
    Trainers.speech_separation, default_args=kwargs)
trainer.model.load_check_point(device=trainer.device)
print(trainer.evaluate(None))
```

MossFormer模型与其它SOTA模型在公开数据集WSJ0-2mix/3mix和WHAM！/WHAMR！上的对比结果如下：

<div align=center>
<img width="640" src="https://modelscope.cn/api/v1/models/damo/speech_mossformer_separation_temporal_8k/repo?Revision=master&FilePath=description/matrix.jpg&View=true"/>
</div>
<center>图2 各模型评估结果</center>

### 指标说明：

* SI-SNR (Scale Invariant Signal-to-Noise Ratio) 尺度不变的信噪比，是在普通信噪比基础上通过正则化消减信号变化导致的影响，是针对宽带噪声失真的语音增强算法的常规衡量方法。SI-SNRi (SI-SNR improvement) 是衡量对比原始混合语音，SI-SNR在分离后语音上的提升量。
* DM (Dynamic Mixing)是一种动态混合数据增强算法，用来补充训练数据的不足和提升模型训练的泛化能力。

#### 相关论文以及引用信息

Zhao, Shengkui and Ma, Bin, “MossFormer: Pushing the Performance Limit of Monaural Speech Separation using Gated Single-head Transformer with Convolution-augmented Joint Self-Attentions”, Accepted by ICASSP 2023.

```BibTeX

```
