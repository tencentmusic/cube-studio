
# Controllable Time-delay Transformer模型介绍

[//]: # (Controllable Time-delay Transformer 模型是一种端到端标点分类模型。)

[//]: # (常规的Transformer会依赖很远的未来信息，导致长时间结果不固定。Controllable Time-delay Transformer 在效果无损的情况下，有效控制标点的延时。)

# Highlights
- 中文标点通用模型：可用于语音识别模型输出文本的标点预测。
  - 基于[Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)场景的使用
  - 基于[FunASR框架](https://github.com/alibaba-damo-academy/FunASR)，可进行ASR，VAD，标点的自由组合
  - 基于纯文本输入的标点预测

## Release Notes
- 2023年3月（3月17号发布）：[funasr-0.2.0](https://github.com/alibaba-damo-academy/FunASR/tree/main)
  - 功能完善：
    - 优化[标点通用模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)，提升模型效果。

- 2023年2月（2月17号发布）：[funasr-0.2.0](https://github.com/alibaba-damo-academy/FunASR/tree/main), modelscope-1.3.0
  - 功能完善：
    - 优化[标点通用模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)，增加标点召回和精度，修复缺少标点等问题。

- 2023年1月（预计1月16号发布）：funasr-0.1.6, modelscope-1.1.4
  - 模型功能完善：
    - Modelscope模型推理pipeline，新增加多种输入音频方式，如wav.scp、音频bytes、音频采样点、WAV格式等。
    - [Paraformer-large模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)，新增加基于ModelScope微调定制模型，新增加batch级解码，加快推理速度。
    - [AISHELL-1学术集Paraformer模型](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)，
    [AISHELL-1学术集ParaformerBert模型](https://modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)，
    [AISHELL-1学术集Conformer模型](https://modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)、
    [AISHELL-2学术集Paraformer模型](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)，
    [AISHELL-2学术集ParaformerBert模型](https://www.modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)、
    [AISHELL-2学术集Conformer模型](https://www.modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)，
    新增加基于ModelScope微调定制模型，其中，Paraformer与ParaformerBert模型新增加batch级解码，加快推理速度。
  - 上线新模型：
    - [Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)，集成VAD、ASR、标点与时间戳功能，可直接对时长为数小时音频进行识别，并输出带标点文字与时间戳。
    - [中文无监督预训练Data2vec模型](https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch/summary)，采用Data2vec结构，基于AISHELL-2数据的中文无监督预训练模型，支持ASR或者下游任务微调模型。
    - [语音端点检查VAD模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)，可用于检测长语音片段中有效语音的起止时间点。
    - [中文标点预测通用模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)，可用于语音识别模型输出文本的标点预测。
    - [8K UniASR流式模型](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online/summary)，[8K UniASR模型](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-offline/summary)，一种流式与离线一体化语音识别模型，进行流式语音识别的同时，能够以较低延时输出离线识别结果来纠正预测文本。
    - Paraformer-large基于[AISHELL-1微调模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell1-vocab8404-pytorch/summary)、[AISHELL-2微调模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell2-vocab8404-pytorch/summary)，将Paraformer-large模型分别基于AISHELL-1与AISHELL-2数据微调。
    - [说话人确认模型](https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary) ，可用于说话人确认，也可以用来做说话人特征提取。
    - [小尺寸设备端Paraformer指令词模型](https://www.modelscope.cn/models/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/summary)，Paraformer-tiny指令词版本，使用小参数量模型支持指令词识别。
  - 将原TensorFlow模型升级为Pytorch模型，进行推理，并支持微调定制，包括：
    - 16K 模型：[Paraformer中文](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)、
    [Paraformer-large中文](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)、
    [UniASR中文](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline/summary)、
    [UniASR-large中文](https://modelscope.cn/models/damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline/summary)、
    [UniASR中文流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-online/summary)、
    [UniASR方言](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-cn-dialect-16k-vocab8358-tensorflow1-offline/summary)、
    [UniASR方言流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-cn-dialect-16k-vocab8358-tensorflow1-online/summary)、
    [UniASR日语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline/summary)、
    [UniASR日语流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online/summary)、
    [UniASR印尼语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-id-16k-common-vocab1067-tensorflow1-offline/summary)、
    [UniASR印尼语流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-id-16k-common-vocab1067-tensorflow1-online/summary)、
    [UniASR葡萄牙语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-pt-16k-common-vocab1617-tensorflow1-offline/summary)、
    [UniASR葡萄牙语流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-pt-16k-common-vocab1617-tensorflow1-online/summary)、
    [UniASR英文](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-offline/summary)、
    [UniASR英文流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-online/summary)、
    [UniASR俄语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ru-16k-common-vocab1664-tensorflow1-offline/summary)、
    [UniASR俄语流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ru-16k-common-vocab1664-tensorflow1-online/summary)、
    [UniASR韩语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline/summary)、
    [UniASR韩语流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-online/summary)、
    [UniASR西班牙语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-es-16k-common-vocab3445-tensorflow1-offline/summary)、
    [UniASR西班牙语流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-es-16k-common-vocab3445-tensorflow1-online/summary)、
    [UniASR粤语简体](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-offline/files)、
    [UniASR粤语简体流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online/files)、
    - 8K 模型：[Paraformer中文](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-8k-common-vocab8358-tensorflow1/summary)、
    [UniASR中文](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab8358-tensorflow1-offline/summary)、
    [UniASR中文流式模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab8358-tensorflow1-offline/summary)

- 2022年11月：funasr-0.1.4, modelscope-1.1.3
  - Paraformer-large非自回归模型上线，多个公开数据集上取得SOTA效果，FunASR框架：
    - 支持基于ModelScope推理。
    - 支持基于[FunASR框架](https://github.com/alibaba-damo-academy/FunASR)微调和推理。

## 项目介绍

Controllable Time-delay Transformer是达摩院语音团队提出的高效后处理框架中的标点模块。本项目为中文通用标点模型，模型可以被应用于文本类输入的标点预测，也可应用于语音识别结果的后处理步骤，协助语音识别模块输出具有可读性的文本结果。

<p align="center">
<img src="fig/struct.png" alt="Controllable Time-delay Transformer模型结构"  width="500" />

Controllable Time-delay Transformer 模型结构如上图所示，由 Embedding、Encoder 和 Predictor 三部分组成。Embedding 是词向量叠加位置向量。Encoder可以采用不同的网络结构，例如self-attention，conformer，SAN-M等。Predictor 预测每个token后的标点类型。

在模型的选择上采用了性能优越的Transformer模型。Transformer模型在获得良好性能的同时，由于模型自身序列化输入等特性，会给系统带来较大时延。常规的Transformer可以看到未来的全部信息，导致标点会依赖很远的未来信息。这会给用户带来一种标点一直在变化刷新，长时间结果不固定的不良感受。基于这一问题，我们创新性的提出了可控时延的Transformer模型（Controllable Time-Delay Transformer, CT-Transformer），在模型性能无损失的情况下，有效控制标点的延时。

更详细的细节见：
- 论文： [CONTROLLABLE TIME-DELAY TRANSFORMER FOR REAL-TIME PUNCTUATION PREDICTION AND DISFLUENCY DETECTION](https://arxiv.org/pdf/2003.01309.pdf)

## 如何使用与训练自己的模型

本项目提供的预训练模型是基于大数据训练的通用领域识别模型，开发者可以基于此模型进一步利用ModelScope的微调功能或者本项目对应的Github代码仓库[FunASR](https://github.com/alibaba-damo-academy/FunASR)进一步进行模型的领域定制化。

### 在Notebook中开发

对于有开发需求的使用者，特别推荐您使用Notebook进行离线处理。先登录ModelScope账号，点击模型页面右上角的“在Notebook中打开”按钮出现对话框，首次使用会提示您关联阿里云账号，按提示操作即可。关联账号后可进入选择启动实例界面，选择计算资源，建立实例，待实例创建完成后进入开发环境，进行调用。


#### 基于ModelScope进行推理

以下为三种支持格式及api调用方式参考如下范例：
- text.scp文件路径，例如example/punc_example.txt，格式为： key + "\t" + value
```sh
cat example/punc_example.txt
1       跨境河流是养育沿岸人民的生命之源
2       从存储上来说仅仅是全景图片它就会是图片的四倍的容量
3       那今天的会就到这里吧happy new year明年见
```
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    model_revision="v1.1.7")

rec_result = inference_pipline(text_in='example/punc_example.txt')
print(rec_result)
```
- text二进制数据，例如：用户直接从文件里读出bytes数据
```python
rec_result = inference_pipline(text_in='我们都是木头人不会讲话不会动')
```
- text文件url，例如：https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt
```python
rec_result = inference_pipline(text_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt')
```


#### 基于ModelScope进行微调
待开发


### 在本地机器中开发

#### 基于ModelScope进行微调和推理

支持基于ModelScope上数据集及私有数据集进行定制微调和推理，使用方式同Notebook中开发。

#### 基于FunASR进行微调和推理

FunASR框架支持魔搭社区开源的工业级的语音识别模型的training & finetuning，使得研究人员和开发者可以更加便捷的进行语音识别模型的研究和生产，目前已在Github开源：https://github.com/alibaba-damo-academy/FunASR

#### FunASR框架安装

- 安装FunASR和ModelScope

```sh
pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/alibaba/FunASR.git
cd FunASR
pip install --editable ./
```


#### 基于FunASR进行推理

接下来会以私有数据集为例，介绍如何在FunASR框架中使用本模型进行推理以及微调。

```sh
cd egs_modelscope/punctuation/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/
python infer.py
```

#### 基于FunASR进行微调
待开发

## Benchmark
中文标点预测通用模型在自采集的通用领域业务场景数据上有良好效果。训练数据大约33M个sample，每个sample可能包含1句或多句。

### 自采集数据（20000+ samples）

| precision                            | recall                                | f1_score                              |
|:------------------------------------:|:-------------------------------------:|:-------------------------------------:|
| <div style="width: 150pt">53.8</div> | <div style="width: 150pt">60.0</div>  | <div style="width: 150pt">56.5</div>  | 

## 使用方式以及适用范围

运行范围
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

使用方式
- 直接推理：可以直接对输入文本进行计算，输出带有标点的目标文字。

使用范围与目标场景
- 适合对文本数据进行标点预测，文本长度不限。

## 相关论文以及引用信息

```BibTeX
@inproceedings{chen2020controllable,
  title={Controllable Time-Delay Transformer for Real-Time Punctuation Prediction and Disfluency Detection},
  author={Chen, Qian and Chen, Mengzhe and Li, Bo and Wang, Wen},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={8069--8073},
  year={2020},
  organization={IEEE}
}
```

