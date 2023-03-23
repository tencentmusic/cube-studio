
# DFSMN回声消除模型介绍

本模型是一种音频通话场景的单通道回声消除模型算法。

## 模型描述

模型接受单通道麦克风信号和单通道参考信号作为输入，输出线性回声消除和回声残余抑制后的音频信号。其中，线性回声消除采用加权的RLS滤波算法，回声残余抑制模型采用Deep FSMN结构。模型的输入是原始观测信号以及线性滤波后信号的Fbank特征，模型的输出是目标语音的Phase senstive mask。模型的训练数据采用[AEC-Challenge](https://github.com/microsoft/AEC-Challenge)开源数据集以及仿真生成的回声数据集。

#### 回声消除应用场景示意：

<div align=center>
<img width="570" src="https://modelscope.cn/api/v1/models/damo/speech_dfsmn_aec_psm_16k/repo?Revision=master&FilePath=description/scenario.jpg&View=true"/>
</div>

#### 模型训练和推理流程示意：

<div align=center>
<img width="630" src="description/model.png"/>
</div>

## 模型的使用方式

模型pipeline 输入为两个16KHz采样率的单声道wav文件，分别是本地麦克风录制信号和远端参考信号，输出结果保存在指定的wav文件中。在安装ModelScope之后，用户还需要做如下环境准备，然后才能使用```speech_dfsmn_aec_psm_16k```进行推理。

#### 运行环境

本模型已针对主流版本Linux，Windows和MacOS系统做过兼容性测试，但不排除在一些旧版本中存在问题，如果您遇到相关错误，请反馈给我们。


#### 代码范例

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


input = {
    'nearend_mic': 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/nearend_mic.wav',
    'farend_speech': 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/farend_speech.wav'
}
aec = pipeline(
   Tasks.acoustic_echo_cancellation,
   model='damo/speech_dfsmn_aec_psm_16k')
result = aec(input, output_path='output.wav')
```

#### 模型局限性

* 由于训练数据偏差，如果麦克风通道存在音乐声，则音乐会被抑制。

* 麦克风和参考通道之间的延迟覆盖范围在500ms以内。

## 数据评估及结果

[AECMOS](https://github.com/microsoft/AEC-Challenge/commit/0cfcae00c08876628d18569313332d4c7446b409) on AEC-Challenge blind_test_set_interspeech2021

| ST NE MOS | ST FE Echo DMOS | DT Echo DMOS | DT Other DMOS |
|:------------:|:---------:|:---------:|:------:|
| 3.04 | 4.44 | 4.70 | 2.59 |

指标说明：

* MOS (Mean Opinion Score) 平均意见得分，是一种主观质量指标，在所有试听人员的评分上求平均得到最终结果，分数范围0-5，越高越好。
* DMOS (Degradation Mean Opinion Score) 失真平均意见分，是应用失真等级评价法 (DCR, Degadation Category Rating)的主观质量指标。

#### 相关论文以及引用信息

```BibTeX
@inproceedings{wang2021weighted,
  title={Weighted recursive least square filter and neural network based residual echo suppression for the aec-challenge},
  author={Wang, Ziteng and Na, Yueyue and Liu, Zhang and Tian, Biao and Fu, Qiang},
  booktitle={2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={141--145},
  year={2021},
  organization={IEEE}
}

@inproceedings{wang2022nn3a,
  title={NN3A: Neural network supported acoustic echo cancellation, noise suppression and automatic gain control for real-time communications},
  author={Wang, Ziteng and Na, Yueyue and Tian, Biao and Fu, Qiang},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={661--665},
  year={2022},
  organization={IEEE}
}
```