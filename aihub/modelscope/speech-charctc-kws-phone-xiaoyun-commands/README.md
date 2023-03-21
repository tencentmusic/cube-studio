
# 语音唤醒模型介绍


## 模型描述

&emsp;&emsp;移动端语音多命令词模型，我们根据以往项目积累，挑选了多个场景常用命令词数据进行模型迭代，所得单一模型支持30+关键词的快速检测：  
&emsp;&emsp;&emsp;&emsp;主唤醒词：小云小云，你好小云  
&emsp;&emsp;&emsp;&emsp;音乐场景命令词：播放音乐，增大音量，减小音量，继续播放，暂停播放，上一首，下一首，单曲循环，随机模式，列表循环  
&emsp;&emsp;&emsp;&emsp;地图场景命令词：取消导航，退出导航，放大地图，查看全程，缩小地图，不走高速，躲避拥堵，避免收费，高速优先  
&emsp;&emsp;&emsp;&emsp;家电场景命令词：返回桌面，睡眠模式，蓝牙模式，打开灯光，关闭灯光，打开空调，关闭空调，拍照拍照，我要拍照  
&emsp;&emsp;&emsp;&emsp;通用场景命令词：上一页，下一页，上一个，下一个，换一批，打开录音，关闭录音  

&emsp;&emsp;模型网络结构继承自[论文](https://www.isca-speech.org/archive/interspeech_2018/chen18c_interspeech.html)《Compact Feedforward Sequential Memory Networks for Small-footprint Keyword Spotting》，其主体为4层cFSMN结构(如下图所示)，参数量约750K，适用于移动端设备运行。  
&emsp;&emsp;模型输入采用Fbank特征，训练阶段使用CTC-loss计算损失并更新参数，输出为基于char建模的中文全集token预测，token数共2599个。测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。  
&emsp;&emsp;模型训练采用"basetrain + finetune"的模式，basetrain过程使用大量内部移动端数据。在此基础上，混合每个命令词数据进行微调，得到最终面向业务的“主唤醒+多命令词”模型。本模型同时训练的命令词个数较多，而内部保有的各命令词数据量差别较大，采集场景各异，最终各选取了3000~30000条录音进行融合训练。  
&emsp;&emsp;由于采用了中文char全量token建模，并使用充分数据进行basetrain，本模型也支持基本的唤醒词/命令词自定义功能，但具体性能无法评估。如用户想验证更多命令词，可以通过页面右侧“在线体验”板块自定义设置并录音测试。  
&emsp;&emsp;目前最新ModelScope版本已支持用户在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型。欢迎您通过[小云小云](https://modelscope.cn/models/damo/speech_charctc_kws_phone-xiaoyun/summary)模型了解唤醒模型定制的方法。  

<p align="center">
<img src="fig/Illustration_of_cFSMN.png" alt="cFSMN网络框图" width="400" />
<p align="left">

## 使用方式和范围

运行范围：
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

使用方式：
- 使用附带的kwsbp工具(Linux-x86_64)直接推理，分别测试各个“主唤醒词/命令词”正样本及负样本集合，根据各个词的Det测试结果综合选取最优工作点。

使用范围:
- 移动端设备，Android/iOS型号或版本不限，使用环境不限，采集音频为16K单通道。

目标场景:
- 移动端APP用到的关键词检测场景。


### 如何使用

#### 模型推理代码范例：

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

kwsbp_16k_pipline = pipeline(
    task=Tasks.keyword_spotting,
    model='damo/speech_charctc_kws_phone-xiaoyun-commands',
    model_revision='v1.0.0')
kws_result = kwsbp_16k_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav')
print(kws_result)
```

audio_in参数说明：
- 默认传入url地址的小云正样本音频，函数返回单条测试结果。
- 设置本地单条音频路径，如audio_in='LOCAL_PATH'，函数返回单条测试结果。
- 设置本地正样本目录(自动检索该目录下wav格式音频)，如audio_in=['POS_DIR', None]，函数返回全部正样本测试结果。
- 设置本地负样本目录(自动检索该目录下wav格式音频)，如audio_in=[None, 'NEG_DIR']，函数返回全部负样本测试结果。
- 同时设置本地正/负样本目录，如audio_in=['POS_DIR', 'NEG_DIR']，函数返回Det测试结果，用户可保存JSON格式文本方便选取合适工作点。


### 模型局限性以及可能的偏差

- 考虑到单一模型同时支持的命令词较多，测试数据缺乏，正负样本测试集覆盖场景不够全面，可能有特定场合/特定人群唤醒率偏低或误唤醒偏高问题。


## 训练数据介绍

- basetrain使用内部移动端ASR数据5000+小时，finetune使用合并37个命令词共约30万条混合场景数据，以及约5万条移动端ASR数据。


## 模型训练流程

- 模型训练采用"basetrain + finetune"的模式，finetune过程使用目标场景的特定唤醒词数据并混合一定比例的ASR数据。如训练数据与应用场景不匹配，应当针对性做数据模拟。


### 预处理

- finetune模型对直接使用各命令词混合场景数据进行训练，未做数据模拟或扩充。


## 数据评估及结果

- 针对“小云小云”主唤醒词，模型在自建9个场景各50句的正样本测试集（共450条），唤醒率为95.78%。  
- 同时从训练数据随机预留10%作为每个命令词的正样本测试集，其中部分词测试数据量较少，可能导致实际体验有偏差；误唤醒依旧使用内部自建的移动端40小时负样本测试集，数据来源于导航场景的ASR。详细测试结果如下：  

<style>
table {
  margin: auto;
}
</style>

|    命令词    |    唤醒率    | 误唤醒/小时 | 正样本(条) | 负样本(小时) |
|:------------:|:------------:|:------------:|:------------:|:------------:|
|  小云小云  |  100.00%  |  0.00   |  1043  |  77.29  |
|  你好小云  |  98.39%  |  0.00   |  1550  |  77.76  |
|  播放音乐  |  89.99%  |  0.01   |  689  |  78.04  |
|  增大音量  |  97.54%  |  0.10   |  1503  |  77.50  |
|  减小音量  |  98.75%  |  0.10   |  1526  |  77.48  |
|  继续播放  |  98.65%  |  0.10   |  2079  |  77.61  |
|  暂停播放  |  98.36%  |  0.10   |  2439  |  77.22  |
|  上一首  |  78.55%  |  1.03   |  3939  |  75.99  |
|  下一首  |  90.45%  |  1.00   |  17504  |  67.95  |
|  单曲循环  |  95.42%  |  0.01   |  153  |  78.60  |
|  随机模式  |  94.85%  |  0.03   |  97  |  78.62  |
|  列表循环  |  89.33%  |  0.00   |  75  |  78.62  |
|  取消导航  |  97.87%  |  0.08   |  1364  |  77.83  |
|  退出导航  |  96.89%  |  0.10   |  2157  |  77.58  |
|  放大地图  |  98.21%  |  0.10   |  1115  |  77.95  |
|  缩小地图  |  98.22%  |  0.10   |  1857  |  77.68  |
|  查看全程  |  96.27%  |  0.08   |  993  |  77.98  |
|  不走高速  |  93.10%  |  0.01   |  58  |  78.64  |
|  躲避拥堵  |  97.86%  |  0.00   |  140  |  78.61  |
|  避免收费  |  96.00%  |  0.00   |  50  |  78.64  |
|  高速优先  |  84.05%  |  0.00   |  163  |  78.58  |
|  上一页  |  92.97%  |  0.10   |  384  |  78.52  |
|  下一页  |  95.96%  |  0.10   |  445  |  78.47  |
|  换一批  |  96.20%  |  0.10   |  368  |  78.53  |
|  返回桌面  |  95.22%  |  0.00   |  335  |  78.52  |
|  睡眠模式  |  99.59%  |  0.00   |  246  |  78.33  |
|  蓝牙模式  |  98.46%  |  0.00   |  259  |  78.32  |
|  拍照拍照  |  97.84%  |  0.03   |  928  |  78.03  |
|  我要拍照  |  98.89%  |  0.08   |  451  |  78.04  |
|  上一个  |  90.45%  |  0.51   |  1403  |  77.18  |
|  下一个  |  97.25%  |  0.50   |  6649  |  71.82  |
|  打开灯光  |  98.48%  |  0.00   |  264  |  78.30  |
|  关闭灯光  |  99.52%  |  0.00   |  210  |  78.38  |
|  打开录音  |  97.82%  |  0.00   |  459  |  78.02  |
|  关闭录音  |  99.04%  |  0.00   |  416  |  78.09  |
|  打开空调  |  87.94%  |  0.00   |  141  |  78.54  |
|  关闭空调  |  87.10%  |  0.00   |  155  |  78.53  |


## 相关论文以及引用信息

```BibTeX
@inproceedings{chen18c_interspeech,
  author={Mengzhe Chen and ShiLiang Zhang and Ming Lei and Yong Liu and Haitao Yao and Jie Gao},
  title={{Compact Feedforward Sequential Memory Networks for Small-footprint Keyword Spotting}},
  year=2018,
  booktitle={Proc. Interspeech 2018},
  pages={2663--2667},
  doi={10.21437/Interspeech.2018-1204}
}
```