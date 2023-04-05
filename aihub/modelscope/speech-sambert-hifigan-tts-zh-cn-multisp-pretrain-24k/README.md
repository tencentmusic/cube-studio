
# Sambert-Hifigan模型介绍

模型体验及训练教程详见:[Sambert-Hifigan模型训练教程](https://modelscope.cn/docs/sambert)

## 框架描述
拼接法和参数法是两种Text-To-Speech(TTS)技术路线。近年来参数TTS系统获得了广泛的应用，故此处仅涉及参数法。

参数TTS系统可分为两大模块：前端和后端。
前端包含文本正则、分词、多音字预测、文本转音素和韵律预测等模块，它的功能是把输入文本进行解析，获得音素、音调、停顿和位置等语言学特征。
后端包含时长模型、声学模型和声码器，它的功能是将语言学特征转换为语音。其中，时长模型的功能是给定语言学特征，获得每一个建模单元（例如:音素）的时长信息；声学模型则基于语言学特征和时长信息预测声学特征；声码器则将声学特征转换为对应的语音波形。

其系统结构如[图1]所示：

![系统结构](description/tts-system.jpg)

前端模块我们采用模型结合规则的方式灵活处理各种场景下的文本，后端模块则采用SAM-BERT + HIFIGAN提供高表现力的流式合成效果。

### 声学模型SAM-BERT
后端模块中声学模型采用自研的SAM-BERT,将时长模型和声学模型联合进行建模。结构如[图2]所示
```
1. Backbone采用Self-Attention-Mechanism(SAM)，提升模型建模能力。
2. Encoder部分采用BERT进行初始化，引入更多文本信息，提升合成韵律。
3. Variance Adaptor对音素级别的韵律(基频、能量、时长)轮廓进行粗粒度的预测，再通过decoder进行帧级别细粒度的建模;并在时长预测时考虑到其与基频、能量的关联信息，结合自回归结构，进一步提升韵律自然度.
4. Decoder部分采用PNCA AR-Decoder[@li2020robutrans]，自然支持流式合成。
```


![SAMBERT结构](description/sambert.jpg)

### 声码器模型:HIFI-GAN
后端模块中声码器采用HIFI-GAN, 基于GAN的方式利用判别器(Discriminator)来指导声码器(即生成器Generator)的训练，相较于经典的自回归式逐样本点CE训练, 训练方式更加自然，在生成效率和效果上具有明显的优势。其系统结构如[图3]所示：

![系统结构](description/hifigan.jpg)

在HIFI-GAN开源工作[1]的基础上，我们针对16k, 48k采样率下的模型结构进行了调优设计，并提供了基于因果卷积的低时延流式生成和chunk流式生成机制，可与声学模型配合支持CPU、GPU等硬件条件下的实时流式合成。

## 使用方式和范围

使用方式：
* 直接输入文本进行推理

使用范围:
* 适用于中文的语音合成场景，输入文本使用utf-8编码，整体长度建议不超过30字

目标场景:
* 各种语音合成任务，比如配音，虚拟主播，数字人等

### 如何使用
目前仅支持Linux使用，暂不支持Windows及Mac使用。

参考代码范例的推理部分可以了解如何使用pipeline进行推理；参考训练部分可以了解如何进行finetune。

目前基于Modelscope框架的训练还不能直接使用Aishell3数据集，需要参考[语音合成-中文-KANTTS-公开数据集](https://modelscope.cn/datasets/speech_tts/speech_kantts_opendata/summary)数据集格式进行调整。

也可以参考[KAN-TTS](https://github.com/AlibabaResearch/KAN-TTS)代码进行基于KAN-TTS框架的finetune。具体使用方法参考：

[sambert训练教程](https://github.com/AlibabaResearch/KAN-TTS/wiki/training_sambert)

[hifigan训练教程](https://github.com/AlibabaResearch/KAN-TTS/wiki/training_hifigan)

使用git克隆预训练模型
```
git clone -b pretrain http://www.modelscope.cn/speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_24k.git
```

#### 代码范例
推理
```Python
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

text = '待合成文本'
model_id = 'speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_24k'
sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model=model_id)
output = sambert_hifigan_tts(input=text)
wav = output[OutputKeys.OUTPUT_WAV]
with open('output.wav', 'wb') as f:
    f.write(wav)
```
训练
```Python
import os
import shutil
import tempfile

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import TtsTrainType

model_id = 'speech_tts/speech_sambert-hifigan_tts_zh-cn_multisp_pretrain_24k'
dataset_id = 'speech_kantts_opendata'
dataset_namespace = 'speech_tts'
# 训练信息，用于指定需要训练哪个或哪些模型，这里展示AM和Vocoder模型皆进行训练
# 目前支持训练：TtsTrainType.TRAIN_TYPE_SAMBERT, TtsTrainType.TRAIN_TYPE_VOC
# 训练SAMBERT会以模型最新step作为基础进行finetune
# 训练Vocoder（HifiGAN）会从0开始进行训练，指定多少个step，训练多少个step
train_info = {
    TtsTrainType.TRAIN_TYPE_SAMBERT: {  # 配置训练AM（sambert）模型
        'train_steps': 2,               # 训练多少个step 
        'save_interval_steps': 1,       # 每训练多少个step保存一次checkpoint
        'eval_interval_steps': 1,       # 每训练多少个step评估一次
        'log_interval': 1               # 每训练多少个step打印一次训练日志
    },
    TtsTrainType.TRAIN_TYPE_VOC: {      # 配置训练Vocoder（HifiGAN）模型
        'train_steps': 2,
        'save_interval_steps': 1,
        'eval_interval_steps': 1,
        'log_interval': 1
    }
}
# 这里展示使用临时目录作为训练的workdir
tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
# 配置训练参数，指定数据集，临时工作目录和train_info
kwargs = dict(
    model=model_id,                             # 指定要finetune的模型
    work_dir=tmp_dir,                           # 指定临时工作目录
    train_dataset=dataset_id,                   # 指定数据集id
    train_dataset_namespace=dataset_namespace,  # 指定数据集所属namespace
    train_type=train_info                       # 指定要训练类型及参数
)                      
trainer = build_trainer(
    Trainers.speech_kantts_trainer, default_args=kwargs)
trainer.train()
# 训练好的checkpoint位于{tmp_dir}/tmp_am/ckpt及{tmp_dir}/tmp_voc/ckpt中
tmp_am = os.path.join(tmp_dir, 'tmp_am', 'ckpt')
tmp_voc = os.path.join(tmp_dir, 'tmp_voc', 'ckpt')
assert os.path.exists(tmp_am)
assert os.path.exists(tmp_voc)
```

### 模型局限性以及可能的偏差
* 该发音人支持中文及英文混合，TN规则为中文


## 训练数据介绍
使用6个不同发音人，共计约100小时数据训练, 主要为中文语料, 包含少量英文语料。

## 模型训练流程
模型所需训练数据格式为：音频(.wav), 文本标注(.txt), 音素时长标注(.interval),  随机初始化训练要求训练数据规模在2小时以上，对于2小时以下的数据集，需使用多人预训练模型进行参数初始化。其中，AM模型训练时间需要1～2天，Vocoder模型训练时间需要5～7天。

### 预处理
模型训练需对音频文件提取声学特征(梅尔频谱)；音素时长根据配置项中的帧长将时间单位转换成帧数；文本标注，根据配置项中的音素集、音调分类、边界分类转换成对应的one-hot编号；

## 数据评估及结果
我们使用MOS（Mean Opinion Score)来评估合成声音的自然度，评分从1（不好）到5（非常好），每提高0.5分表示更高的自然度。我们会随机选择20个samples，然后每个sample交给至少10个人进行打分。作为对比，我们会使用真人录音的sample通过上述统计方式进行打分。


|    MOS    | angry | fear  | happy | hate  | neural |  sad  | surprise | average |
|:---------:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|:--------:|:-------:|
| recording | 4.622 | 4.609 | 4.681 | 4.523 | 4.539  | 4.648 |  4.691   | 4.6161  |
| synthesis | 4.601 | 4.658 | 4.549 | 4.614 | 4.466  | 4.691 |  4.542   | 4.5887  |


## 引用
如果你觉得这个该模型对有所帮助，请考虑引用下面的相关的论文：

```BibTeX
@inproceedings{li2020robutrans,
  title={Robutrans: A robust transformer-based text-to-speech model},
  author={Li, Naihan and Liu, Yanqing and Wu, Yu and Liu, Shujie and Zhao, Sheng and Liu, Ming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  pages={8228--8235},
  year={2020}
}
```

```BibTeX
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```
```BibTeX
@article{kong2020hifi,
  title={Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis},
  author={Kong, Jungil and Kim, Jaehyeon and Bae, Jaekyoung},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={17022--17033},
  year={2020}
}
```

本模型参考了以下实现
- [1] [ming024's FastSpeech2 Implementation](https://github.com/ming024/FastSpeech2)
- [2] [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [3] [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [4] [mozilla/TTS](https://github.com/mozilla/TTS)
- [5] [espnet/espnet](https://github.com/espnet/espnet)



