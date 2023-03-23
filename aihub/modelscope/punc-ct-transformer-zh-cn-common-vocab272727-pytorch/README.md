
# Controllable Time-delay Transformer模型介绍

[//]: # (Controllable Time-delay Transformer 模型是一种端到端标点分类模型。)

[//]: # (常规的Transformer会依赖很远的未来信息，导致长时间结果不固定。Controllable Time-delay Transformer 在效果无损的情况下，有效控制标点的延时。)

# Highlights
- 中文标点通用模型：可用于语音识别模型输出文本的标点预测。
  - 基于[Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)场景的使用
  - 基于[FunASR框架](https://github.com/alibaba-damo-academy/FunASR)，可进行ASR，VAD，标点的自由组合
  - 基于纯文本输入的标点预测

## Release Notes
- 2023年3月17日：[funasr-0.3.0](https://github.com/alibaba-damo-academy/FunASR/tree/main), modelscope-1.4.1
  - 功能完善：
    - 新增GPU runtime方案，[nv-triton](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/triton_gpu)，可以将modelscope中Paraformer模型便捷导出，并部署成triton服务，实测，单GPU-V100，RTF为0.0032，吞吐率为300，[benchmark](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/triton_gpu#performance-benchmark)。
    - 新增CPU [runtime量化方案](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)，支持从modelscope导出量化版本onnx与libtorch，实测，CPU-8369B，量化后，RTF提升50%（0.00438->0.00226），吞吐率翻倍（228->442），[benchmark](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python)。
    - [新增加C++版本grpc服务部署方案](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/grpc)，配合C++版本[onnxruntime](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/onnxruntime)，以及[量化方案](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)，相比python-runtime性能翻倍。
    - [16k VAD模型](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)，[8k VAD模型](https://www.modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-8k-common/summary)，modelscope pipeline，新增加流式推理方式，，最小支持10ms语音输入流，[用法](https://github.com/alibaba-damo-academy/FunASR/discussions/236)。
    - 优化[标点预测模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)，主观体验标点准确性提升(fscore绝对提升 55.6->56.5)。
    - 基于grpc服务，新增实时字幕[demo](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/grpc)，采用2pass识别模型，[Paraformer流式模型](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary) 用来上屏，[Paraformer-large离线模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)用来纠正识别结果。
  - 上线新模型：
    - [16k Paraformer流式模型](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary)，支持语音流输入，可以进行实时语音识别，[用法](https://github.com/alibaba-damo-academy/FunASR/discussions/241)。支持基于grpc服务进行部署，可实现实时字幕功能。
    - [流式标点模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/summary)，支持流式语音识别场景中的标点打标，以VAD点为实时调用点进行流式调用。可与实时ASR模型配合使用，实现具有可读性的实时字幕功能，[用法](https://github.com/alibaba-damo-academy/FunASR/discussions/238)
    - [TP-Aligner时间戳模型](https://www.modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary)，输入音频及对应文本输出字级别时间戳，效果与Kaldi FA模型相当（60.3ms v.s. 69.3ms），支持与asr模型自由组合，[用法](https://github.com/alibaba-damo-academy/FunASR/discussions/246)。
    - 金融领域模型，[8k Paraformer-large-3445vocab](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-8k-finance-vocab3445/summary)，使用1000小时数据微调训练，金融领域测试集识别效果相对提升5%，领域关键词召回相对提升7%。
    - 音视频领域模型，[16k Paraformer-large-3445vocab](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-audio_and_video-vocab3445/summary)，使用10000小时数据微调训练，音视频领域测试集识别效果相对提升8%。
    - [8k说话人确认模型](https://www.modelscope.cn/models/damo/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch/summary)，CallHome数据集英文说话人确认模型，也可用于声纹特征提取。
    - 说话人日志模型，[16k SOND中文模型](https://www.modelscope.cn/models/damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch/summary)，[8k SOND英文模型](https://www.modelscope.cn/models/damo/speech_diarization_sond-en-us-callhome-8k-n16k4-pytorch/summary)，在AliMeeting和Callhome上获得最优性能，DER分别为4.46%和11.13%。
    - UniASR流式离线一体化模型: 
      [16k UniASR缅甸语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-my-16k-common-vocab696-pytorch/summary)、[16k UniASR希伯来语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-he-16k-common-vocab1085-pytorch/summary)、[16k UniASR乌尔都语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ur-16k-common-vocab877-pytorch/summary)、[8k UniASR中文金融领域](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-finance-vocab3445-online/summary)、[16k UniASR中文音视频领域](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-audio_and_video-vocab3445-online/summary)。

- 历史 Release Notes，[详细版本](https://github.com/alibaba-damo-academy/FunASR/releases)
  - 重点模型如下：
    - [MFCCA多通道多说话人识别模型](https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary)
    - 标点模型：
    [中文标点预测通用模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)
    - 说话人确认模型：
    [说话人确认模型](https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary)
    - VAD模型：
    [16k语音端点检测VAD模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)、
    [8k语音端点检测VAD模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-8k-common/summary)
    - Paraformer离线模型：
    [16k Paraformer-large中英文模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)、
    [16k Paraformer-large热词模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary)、
    [16k Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)、
    [16k Paraformer中文](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)、
    [16k Paraformer-large中文](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)、
    [8k Paraformer中文](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-8k-common-vocab8358-tensorflow1/summary)、
    [小尺寸设备端Paraformer指令词模型](https://www.modelscope.cn/models/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/summary)
    - UniASR流式离线一体化模型: 
    [UniASR中文模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-online/summary)、
    [UniASR方言模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-cn-dialect-16k-vocab8358-tensorflow1-online/summary)、
    [16k UniASR闽南语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-minnan-16k-common-vocab3825/summary)、
    [16k UniASR法语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fr-16k-common-vocab3472-tensorflow1-online/summary)、
    [16k UniASR德语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-de-16k-common-vocab3690-tensorflow1-online/summary)、
    [16k UniASR越南语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online/summary)、
    [16k UniASR波斯语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fa-16k-common-vocab1257-pytorch-online/summary)。
    [16k UniASR-large中文](https://modelscope.cn/models/damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline/summary)、
    [16k UniASR日语模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online/summary)、
    [16k UniASR印尼语模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-id-16k-common-vocab1067-tensorflow1-online/summary)、
    [16k UniASR葡萄牙语模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-pt-16k-common-vocab1617-tensorflow1-online/summary)、
    [16k UniASR英文模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-online/summary)、
    [16k UniASR俄语模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ru-16k-common-vocab1664-tensorflow1-online/summary)、
    [16k UniASR韩语模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-online/summary)、
    [16k UniASR西班牙语模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-es-16k-common-vocab3445-tensorflow1-online/summary)、
    [16k UniASR粤语简体模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online/files)、
    [8k UniASR中文-vocab8358](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab8358-tensorflow1-offline/summary)、
    [8K UniASR流式模型](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online/summary)

    - 无监督预训练模型：
    [中文无监督预训练Data2vec模型](https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch/summary)、
    [基于Data2vec结构无监督预训练Paraformer模型](https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-paraformer-zh-cn-aishell2-16k/summary)。

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

