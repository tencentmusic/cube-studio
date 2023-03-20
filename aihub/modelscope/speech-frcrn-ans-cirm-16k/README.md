

# FRCRN语音降噪模型介绍

我们日常可能会碰到一些录音质量不佳的场景。比如，想录制一段干净的语音却发现周围都很吵，录制的语音里往往混杂着噪声。当我们在噪杂的地铁或者巴士上通电话，为了让对方听清楚，不得不提高嗓门和音量。这都是因为环境噪声的影响，使我们在使用语音应用时出现障碍。这是语音通讯中一个普遍存在且又非常棘手的问题。语音质量（quality）和可懂度（intelligibility）容易受到环境噪声、拾音设备、混响及回声的干扰，使通话质量和交流效率大幅降低，如何在嘈杂的环境中保持较高的语音质量和可懂度一直以来是众多企业和学者追求的目标。

语音降噪问题通过多年研发积累，已经取得一定的突破，尤其针对复杂环境中的语音降噪问题，通过融入复数域深度学习算法，在性能上获得大幅度的提升，在保障更小语音失真度的情况下，最大限度地消除背景噪声，还原目标语音的清晰度，因而语音降噪模型也通常被叫做语音增强模型。

语音降噪模型的作用是从污染的语音中提取目标语音，还原目标语音质量和可懂度，同时提升语音识别的效果和性能。我们的语音降噪模型只需要输入单麦克风的录音音频，便能够输出降噪后的干净语音音频，即保持音频的格式不变，仅消除音频中的噪声和混响部分，最大限度地保留原始语音。

## 模型描述

FRCRN语音降噪模型是基于频率循环 CRN (FRCRN) 新框架开发出来的。该框架是在卷积编-解码(Convolutional Encoder-Decoder)架构的基础上，通过进一步增加循环层获得的卷积循环编-解码(Convolutional Recurrent Encoder-Decoder)新型架构，可以明显改善卷积核的视野局限性，提升降噪模型对频率维度的特征表达，尤其是在频率长距离相关性表达上获得提升，可以在消除噪声的同时，对语音进行更针对性的辨识和保护。

另外，我们引入前馈序列记忆网络（Feedforward Sequential Memory Network: FSMN）来降低循环网络的复杂性，以及结合复数域网络运算，实现全复数深度网络模型算法，不仅更有效地对长序列语音进行建模，同时对语音的幅度和相位进行同时增强，相关模型在IEEE/INTERSpeech DNS Challenge上有较好的表现。本次开放的模型在参赛版本基础上做了进一步优化，使用了两个Unet级联和SE layer，可以获得更为稳定的效果。如果用户需要因果模型，也可以自行修改代码，把模型中的SElayer替换成卷积层或者加上掩蔽即可。

该模型神经网络结构如下图所示。

![model.png](description/model.png)

模型输入和输出均为16kHz采样率单通道语音时域波形信号，输入信号可由单通道麦克风直接进行录制，输出为噪声抑制后的语音音频信号[1]。模型输入信号通过STFT变换转换成复数频谱特征作为输入，并采用Complex FSMN在频域上进行关联性处理和在时序特征上进行长序处理，预测中间输出目标Complex ideal ratio mask, 然后使用预测的mask和输入频谱相乘后得到增强后的频谱，最后通过STFT逆变换得到增强后语音波形信号。

## 期望模型使用方式以及适用范围


### 如何使用

在安装ModelScope完成之后即可使用```speech_frcrn_ans_cirm_16k```进行推理。模型输入和输出均为16kHz采样率单通道语音时域波形信号，输入信号可由单通道麦克风直接进行录制，输出为噪声抑制后的语音音频信号。为了方便使用在pipeline在模型处理前后增加了wav文件处理逻辑，可以直接读取一个wav文件，并把输出结果保存在指定的wav文件中。

#### 环境准备：

* 本模型支持Linxu，Windows和MacOS平台。
* 本模型已经在1.8~1.11和1.13 下测试通过，由于PyTorch v1.12的[BUG](https://github.com/pytorch/pytorch/issues/80837)，无法在v1.12上运行，请升级到新版或执行以下命令回退到v1.11

```
conda install pytorch==1.11 torchaudio torchvision -c pytorch
```

* 本模型的pipeline中使用了三方库SoundFile进行wav文件处理，**在Linux系统上用户需要手动安装SoundFile的底层依赖库libsndfile**，在Windows和MacOS上会自动安装不需要用户操作。详细信息可参考[SoundFile官网](https://github.com/bastibe/python-soundfile#installation)。以Ubuntu系统为例，用户需要执行如下命令:

```shell
sudo apt-get update
sudo apt-get install libsndfile1
```

#### 代码范例

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')
result = ans(
    'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise1.wav',
    output_path='output.wav')
```

### 模型局限性以及可能的偏差

模型在存在多说话人干扰声的场景噪声抑制性能有不同程度的下降。

## 训练数据介绍

模型的训练数据来自DNS-Challenge开源数据集，是Microsoft团队为ICASSP相关挑战赛提供的，[官方网址](https://github.com/microsoft/DNS-Challenge)[2]。我们这个模型是用来处理16k音频，因此只使用了其中的fullband数据，并做了少量调整。为便于大家使用，我们把DNS Challenge 2020的数据集迁移在modelscope的[DatasetHub](https://modelscope.cn/datasets/modelscope/ICASSP_2021_DNS_Challenge/summary)上，用户可参照数据集说明文档下载使用。

## 模型训练流程

### 复制官方模型
要训练您自己的降噪模型，首先需要一份官方模型的副本。ModelScope 框架默认把官方模型保存在本地缓存中，可以把本地缓存的模型目录copy一份到您的工作目录。

检查目录./speech_frcrn_ans_cirm_16k，其中的 pytorch_model.bin 就是模型文件。如果想从头开始训练一个全新的模型，请删除掉这里的 pytorch_model.bin，避免程序运行时加载；如果想基于官方模型继续训练则不要删除。

```bash
cp -r ~/.cache/modelscope/hub/damo/speech_frcrn_ans_cirm_16k ./
cd ./speech_frcrn_ans_cirm_16k
rm pytorch_model.bin
```

目录中的configuration.json文件中是模型和训练的配置项，建议用户对代码逻辑非常熟悉以后再尝试修改。

### 运行训练代码

以下列出的为训练示例代码，其中有两个地方需要替换成您的本地路径：

1. 用您前面下载的本地数据集路径替换`/your_local_path/ICASSP_2021_DNS_Challenge`
2. 用您复制的官方模型路径替换模型路径

```python
import os

from datasets import load_dataset

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

tmp_dir = './checkpoint'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

hf_ds = load_dataset(
    '/your_local_path/ICASSP_2021_DNS_Challenge',
    'train',
    split='train')
mapped_ds = hf_ds.map(
    to_segment,
    remove_columns=['duration'],
    num_proc=8,
    batched=True,
    batch_size=36)
mapped_ds = mapped_ds.train_test_split(test_size=3000)
mapped_ds = mapped_ds.shuffle()
dataset = MsDataset.from_hf_dataset(mapped_ds)

kwargs = dict(
    model='your_local_path/speech_frcrn_ans_cirm_16k',
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    work_dir=tmp_dir)
trainer = build_trainer(
    Trainers.speech_frcrn_ans_cirm_16k, default_args=kwargs)
trainer.train()
```

训练按照默认配置共200轮，每轮2000个batch，训练出的模型文件会保存在代码中tmp_dir = './checkpoint'指定的目录。目录下还有一个log文件，记录了每个模型的训练和测试loss数据。

### 使用您的模型

从您训练出的模型中选择效果最好的，把模型文件copy到 `/your_local_path/speech_frcrn_ans_cirm_16k` ，重命名为 `pytorch_model.bin` 。
把以下代码中模型路径 `/your_local_path/speech_frcrn_ans_cirm_16k` 替换为您复制的模型目录，就可以测试您的模型效果了。

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='/your_local_path/speech_frcrn_ans_cirm_16k')
result = ans(
    'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/speech_with_noise.wav',
    output_path='output.wav')
```

代码中的http地址也可以换成您的本地音频文件路径，注意模型支持的音频格式是采样率16000，16bit的单通道wav文件。也可参考「在Notebook中开发」章节进行批量的音频文件处理。

## 数据评估及结果

与其他SOTA模型在DNS Challenge 2020官方测试集上对比效果如下：

![matrix.png](description/matrix.png)

指标说明：

* PESQ (Perceptual Evaluation Of Speech Quality) 语音质量感知评估，是一种客观的、全参考的语音质量评估方法，得分范围在-0.5--4.5之间，得分越高表示语音质量越好。
* STOI (Short-Time Objective Intelligibility) 短时客观可懂度，反映人类的听觉感知系统对语音可懂度的客观评价，STOI 值介于0~1 之间，值越大代表语音可懂度越高，越清晰。
* SI-SNR (Scale Invariant Signal-to-Noise Ratio) 尺度不变的信噪比，是在普通信噪比基础上通过正则化消减信号变化导致的影响，是针对宽带噪声失真的语音增强算法的常规衡量方法。

DNS Challenge的结果列表在[这里](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-icassp-2022/results/)。

### 模型评估代码
可通过如下代码对模型进行评估验证，我们在modelscope的[DatasetHub](https://modelscope.cn/datasets/modelscope/ICASSP_2021_DNS_Challenge/summary)上存储了DNS Challenge 2020的验证集，方便用户下载调用。

```python
import os
import tempfile

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

hf_ds = MsDataset.load(
    'ICASSP_2021_DNS_Challenge', split='test').to_hf_dataset()
mapped_ds = hf_ds.map(
    to_segment,
    remove_columns=['duration'],
    # num_proc=5, # Comment this line to avoid error in Jupyter notebook
    batched=True,
    batch_size=36)
dataset = MsDataset.from_hf_dataset(mapped_ds)
kwargs = dict(
    model='damo/speech_frcrn_ans_cirm_16k',
    model_revision='beta',
    train_dataset=None,
    eval_dataset=dataset,
    val_iters_per_epoch=125,
    work_dir=tmp_dir)

trainer = build_trainer(
    Trainers.speech_frcrn_ans_cirm_16k, default_args=kwargs)

eval_res = trainer.evaluate()
print(eval_res['avg_sisnr'])

```

更多详情请参考下面相关论文。

### 相关论文以及引用信息

[1]

```BibTeX
@INPROCEEDINGS{9747578,
  author={Zhao, Shengkui and Ma, Bin and Watcharasupat, Karn N. and Gan, Woon-Seng},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={FRCRN: Boosting Feature Representation Using Frequency Recurrence for Monaural Speech Enhancement}, 
  year={2022},
  pages={9281-9285},
  doi={10.1109/ICASSP43922.2022.9747578}}
```

[2]

```BibTeX
@INPROCEEDINGS{9747230,
  author={Dubey, Harishchandra and Gopal, Vishak and Cutler, Ross and Aazami, Ashkan and Matusevych, Sergiy and Braun, Sebastian and Eskimez, Sefik Emre and Thakker, Manthan and Yoshioka, Takuya and Gamper, Hannes and Aichner, Robert},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Icassp 2022 Deep Noise Suppression Challenge}, 
  year={2022},
  pages={9271-9275},
  doi={10.1109/ICASSP43922.2022.9747230}}
```
