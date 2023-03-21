
# Paraformer-large-热词版模型介绍

[//]: # (Paraformer 模型是一种非自回归（Non-autoregressive）端到端语音识别模型。)

[//]: # (非自回归模型相比于自回归模型，可以对整条句子并行输出目标文字，具有更高的计算效率，尤其采用GPU解码。)

[//]: # (Paraformer模型相比于其他非自回归模型，不仅具有高效的解码效率，在模型参数可比的情况下，模型识别性能与SOTA的自回归模型相当。)

[//]: # (目前Paraformer在如下数据集进行了性能验证：[AISHELL-1]&#40;http://www.openslr.org/33/&#41;、[AISHELL-2]&#40;https://www.aishelltech.com/aishell_2&#41;、[WenetSpeech]&#40;http://www.openslr.org/121/&#41;、阿里内部工业大数据。)


## Highlights
Paraformer-large热词版模型支持热词定制功能：实现热词定制化功能，基于提供的热词列表进行激励增强，提升热词的召回率和准确率。
- [Parformer-large模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)：非自回归语音识别模型，多个中文公开数据集上取得SOTA效果，可快速地基于ModelScope对模型进行微调定制和推理。
- [Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)：集成VAD、ASR、标点与时间戳功能，可直接对时长为数小时音频进行识别，并输出带标点文字与时间戳。


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

[//]: # (  - [Paraformer-large长音频模型]&#40;https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary&#41;，集成VAD、ASR、标点与时间戳功能，可直接对时长为数小时音频进行识别，并输出带标点文字与时间戳。)
[//]: # (  - [中文无监督预训练Data2vec模型]&#40;https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch/summary&#41;，采用Data2vec结构，基于AISHELL-2数据的中文无监督预训练模型，支持ASR或者下游任务微调模型。)

[//]: # (  - [16k语音端点检测VAD模型]&#40;https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary&#41;，可用于检测长语音片段中有效语音的起止时间点。)
[//]: # (  - [中文标点预测通用模型]&#40;https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary&#41;，可用于语音识别模型输出文本的标点预测。)
[//]: # (  - [8K UniASR流式模型]&#40;https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online/summary&#41;，[8K UniASR模型]&#40;https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-offline/summary&#41;，一种流式与离线一体化语音识别模型，进行流式语音识别的同时，能够以较低延时输出离线识别结果来纠正预测文本。)
[//]: # (  - Paraformer-large基于[AISHELL-1微调模型]&#40;https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell1-vocab8404-pytorch/summary&#41;、[AISHELL-2微调模型]&#40;https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell2-vocab8404-pytorch/summary&#41;，将Paraformer-large模型分别基于AISHELL-1与AISHELL-2数据微调。)

[//]: # (  - [说话人确认模型]&#40;https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary&#41; ，可用于说话人确认，也可以用来做说话人特征提取。)

[//]: # (  - [小尺寸设备端Paraformer指令词模型]&#40;https://www.modelscope.cn/models/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/summary&#41;，Paraformer-tiny指令词版本，使用小参数量模型支持指令词识别。)
[//]: # (- 将原TensorFlow模型升级为Pytorch模型，进行推理，并支持微调定制，包括：)

[//]: # (  - 16K 模型：[Paraformer中文]&#40;https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary&#41;、)

[//]: # (  [Paraformer-large中文]&#40;https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary&#41;、)

[//]: # (  - 8K 模型：[Paraformer中文]&#40;https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-8k-common-vocab8358-tensorflow1/summary&#41;、)

[//]: # (  - 功能完善：)

[//]: # (    - Modelscope模型推理pipeline，Paraformer模型新增加batch级解码；新增加多种输入音频方式，如wav.scp、音频bytes、音频采样点、WAV格式等。)

[//]: # (    - 新增加基于ModelScope微调定制pipeline，其中，，加快推理速度。)

[//]: # (    - [Paraformer-large模型]&#40;https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary&#41;，新增加基于ModelScope微调定制模型，新增加batch级解码，加快推理速度。)

[//]: # (    - [AISHELL-1学术集Paraformer模型]&#40;https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary&#41;，)

[//]: # (    [AISHELL-1学术集ParaformerBert模型]&#40;https://modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary&#41;，)

[//]: # (    [AISHELL-1学术集Conformer模型]&#40;https://modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary&#41;、)

[//]: # (    [AISHELL-2学术集Paraformer模型]&#40;https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary&#41;，)

[//]: # (    [AISHELL-2学术集ParaformerBert模型]&#40;https://www.modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary&#41;、)

[//]: # (    [AISHELL-2学术集Conformer模型]&#40;https://www.modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary&#41;，)
[//]: # (    新增加基于ModelScope微调定制模型，其中，Paraformer与ParaformerBert模型新增加batch级解码，加快推理速度。)



## 项目介绍

Paraformer是达摩院语音团队提出的一种高效的非自回归端到端语音识别框架。本项目为Paraformer中文通用语音识别模型，采用工业级数万小时的标注音频进行模型训练，保证了模型的通用识别效果。模型可以被应用于语音输入法、语音导航、智能会议纪要等场景。

<p align="center">
<img src="fig/struct.png" alt="Paraformer模型结构"  width="500" />


Paraformer模型结构如上图所示，由 Encoder、Predictor、Sampler、Decoder 与 Loss function 五部分组成。Encoder可以采用不同的网络结构，例如self-attention，conformer，SAN-M等。Predictor 为两层FFN，预测目标文字个数以及抽取目标文字对应的声学向量。Sampler 为无可学习参数模块，依据输入的声学向量和目标向量，生产含有语义的特征向量。Decoder 结构与自回归模型类似，为双向建模（自回归为单向建模）。Loss function 部分，除了交叉熵（CE）与 MWER 区分性优化目标，还包括了 Predictor 优化目标 MAE。


其核心点主要有：  
- Predictor 模块：基于 Continuous integrate-and-fire (CIF) 的 预测器 (Predictor) 来抽取目标文字对应的声学特征向量，可以更加准确的预测语音中目标文字个数。  
- Sampler：通过采样，将声学特征向量与目标文字向量变换成含有语义信息的特征向量，配合双向的 Decoder 来增强模型对于上下文的建模能力。  
- 基于负样本采样的 MWER 训练准则。  

更详细的细节见：
- 论文： [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition](https://arxiv.org/abs/2206.08317)
- 论文解读：[Paraformer: 高识别率、高计算效率的单轮非自回归端到端语音识别模型](https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw)


## 如何使用与训练自己的模型

本项目提供的预训练模型是基于大数据训练的通用领域识别模型，开发者可以基于此模型进一步利用ModelScope的微调功能或者本项目对应的Github代码仓库[FunASR](https://github.com/alibaba-damo-academy/FunASR)进一步进行模型的领域定制化。

### 在Notebook中开发

对于有开发需求的使用者，特别推荐您使用Notebook进行离线处理。先登录ModelScope账号，点击模型页面右上角的“在Notebook中打开”按钮出现对话框，首次使用会提示您关联阿里云账号，按提示操作即可。关联账号后可进入选择启动实例界面，选择计算资源，建立实例，待实例创建完成后进入开发环境，进行调用。

#### 基于ModelScope进行推理

- 推理支持音频格式如下：
  - wav文件路径，例如：data/test/audios/asr_example_hotword.wav
  - pcm文件路径，例如：data/test/audios/asr_example_hotword.pcm
  - wav文件url，例如：https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_hotword.wav
  - wav二进制数据，格式bytes，例如：用户直接从文件里读出bytes数据或者是麦克风录出bytes数据。
  - 已解析的audio音频，例如：audio, rate = soundfile.read("asr_example_hotword.wav")，类型为numpy.ndarray或者torch.Tensor。
  - wav.scp文件，需符合如下要求：

```sh
cat wav.scp
asr_example1  data/test/audios/asr_example1.wav
asr_example2  data/test/audios/asr_example2.wav
...
```

- 推理支持热词文件格式如下：
  - 字符串str，以空格为间隔，例如：param_dict['hotword']="邓郁松 王颖春 王晔君"
  - txt文件路径，文件中每行包含一个热词，例如：param_dict['hotword']="data/test/hotword.txt"
  - txt文件url，文件中每行包含一个热词，例如：param_dict['hotword']="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/hotword.txt"


- 若输入格式wav文件url，api调用方式可参考如下范例：

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

param_dict = dict()
param_dict['hotword'] = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/hotword.txt"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    param_dict=param_dict)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_hotword.wav')
print(rec_result)
```

- 输入音频为pcm格式，调用api时需要传入音频采样率参数audio_fs，例如：

```python
rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_hotword.pcm', audio_fs=16000)
```

- 若输入格式为文件wav.scp(注：文件名需要以.scp结尾)，可添加 output_dir 参数将识别结果写入文件中，并可设置解码batch_size，参考示例如下：

```python
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    param_dict=param_dict,
    output_dir='./output_dir',
    batch_size=32)

inference_pipeline(audio_in="wav.scp")
```
识别结果输出路径结构如下：

```sh
tree output_dir/
output_dir/
└── 1best_recog
    ├── rtf
    ├── score
    └── text

1 directory, 3 files
```
rtf：计算过程耗时统计

score：识别路径得分

text：语音识别结果文件

- 若输入音频为已解析的audio音频，api调用方式可参考如下范例：

```python
import soundfile

waveform, sample_rate = soundfile.read("asr_example_hotword.wav")
rec_result = inference_pipeline(audio_in=waveform)
```

- AISHELL-1热词测试集效果测试

我们从学术数据集AISHELL-1的测试(test)集中，抽取了235条包含实体词的音频，其中实体词187个，组成了AISHELL-1的test子集作为热词测试集，并上传到ModelScope的[中文语音识别Aishell-1学术数据集热词测试集](https://modelscope.cn/datasets/speech_asr/speech_asr_aishell1_hotwords_testsets/summary)，来体验定制化热词的效果，代码示例如下：

```python
import os
import tempfile
import codecs
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.msdatasets import MsDataset

if __name__ == '__main__':
    param_dict = dict()
    param_dict['hotword'] = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/hotword.txt"

    output_dir = "./output"
    batch_size = 1

    # dataset split ['test']
    ds_dict = MsDataset.load(dataset_name='speech_asr_aishell1_hotwords_testsets', namespace='speech_asr')
    work_dir = tempfile.TemporaryDirectory().name
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    wav_file_path = os.path.join(work_dir, "wav.scp")

    with codecs.open(wav_file_path, 'w') as fin:
        for line in ds_dict:
            wav = line["Audio:FILE"]
            idx = wav.split("/")[-1].split(".")[0]
            fin.writelines(idx + " " + wav + "\n")
    audio_in = wav_file_path

    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
        output_dir=output_dir,
        batch_size=batch_size,
        param_dict=param_dict)

    rec_result = inference_pipeline(audio_in=audio_in)
```

- ASR、VAD、PUNC模型自由组合

可根据使用需求对VAD和PUNC标点模型进行自由组合，使用方式如下：
```python
param_dict = dict()
param_dict['hotword'] = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/hotword.txt"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404',
    vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    vad_model_revision="v1.1.8",
    punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    param_dict=param_dict,
)
```
若不使用PUNC模型，可配置punc_model=""，或不传入punc_model参数，如需加入LM模型，可增加配置lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch'>。

长音频版本模型默认开启时间戳，若不使用时间戳，可通过传入参数param_dict['use_timestamp'] = False关闭时间戳，使用方式如下：
```python
param_dict['use_timestamp'] = False
rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_vad_punc_example.wav', param_dict=param_dict)

)
```


### 在本地机器中开发

#### 基于ModelScope进行推理

本地支持基于ModelScope推理，使用方式同Notebook中开发。

#### 基于FunASR进行推理

FunASR框架支持魔搭社区开源的工业级的语音识别模型的training & finetuning，使得研究人员和开发者可以更加便捷的进行语音识别模型的研究和生产，目前已在Github开源：https://github.com/alibaba-damo-academy/FunASR 。若在使用过程中遇到任何问题，欢迎联系我们：[联系方式](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/images/dingding.jpg)

#### FunASR框架安装

- 安装FunASR和ModelScope

```sh
pip install "modelscope[audio_asr]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/alibaba/FunASR.git
cd FunASR
pip install --editable ./
```

#### 基于FunASR进行推理

接下来会以私有数据集为例，介绍如何在FunASR框架中使用Paraformer-large热词模型进行推理。

```sh
cd egs_modelscope/asr/paraformer/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404
# 配置infer.py中的待识别音频(audio_in)、热词列表(param_dict['hotword'])、输出路径(output_dir)、batch_size参数后，执行：
python infer.py
```

## Benchmark
  结合大数据、大模型优化的Paraformer在一序列语音识别的benchmark上获得当前SOTA的效果，以下展示学术数据集AISHELL-1、AISHELL-2、WenetSpeech，公开评测项目SpeechIO TIOBE白盒测试场景的效果。在学术界常用的中文语音识别评测任务中，其表现远远超于目前公开发表论文中的结果，远好于单独封闭数据集上的模型。

### 实体词测试集

  我们从学术数据集AISHELL-1的测试(test)集中，抽取了235条包含实体词的音频，其中实体词187个，组成了AISHELL-1的test子集作为热词测试集，并已经上传到ModelScope的[中文语音识别Aishell-1学术数据集热词测试集](https://www.modelscope.cn/datasets/speech_asr/speech_asr_aishell1_hotwords_testsets/summary)。在test子集上比较了无热词&热词时模型的效果，执行代码可参考FunASR框架中[infer_aishell1_subtest_demo.py](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr/paraformer/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/infer_aishell1_subtest_demo.py)文件：
| AISHELL-1 test                          | CER                                   |Recall                                 |Precision                              |F1-score                                |
|:------------------------------------------------:|:-------------------------------------:|:-------------------------------------:|:-------------------------------------:|:--------------------------------------:|
| <div style="width: 150pt">test子集 without 热词 </div>    | <div style="width: 150pt">10.01</div> | <div style="width: 150pt">0.16</div>  | <div style="width: 150pt">1.0</div>   | <div style="width: 150pt">0.27</div>   |
| <div style="width: 150pt">test子集 with 热词</div>        | <div style="width: 150pt">4.55</div>  | <div style="width: 150pt">0.74</div>  | <div style="width: 150pt">1.0</div>   | <div style="width: 150pt">0.85</div>   |



### AISHELL-1

| AISHELL-1 test                                   | w/o LM                                | w/ LM                                 |
|:------------------------------------------------:|:-------------------------------------:|:-------------------------------------:|
| <div style="width: 150pt">Espnet</div>           | <div style="width: 150pt">4.90</div>  | <div style="width: 150pt">4.70</div>  | 
| <div style="width: 150pt">Wenet</div>            | <div style="width: 150pt">4.61</div>  | <div style="width: 150pt">4.36</div>  | 
| <div style="width: 150pt">K2</div>               | <div style="width: 150pt">-</div>     | <div style="width: 150pt">4.26</div>  | 
| <div style="width: 150pt">Blockformer</div>      | <div style="width: 150pt">4.29</div>  | <div style="width: 150pt">4.05</div>  |
| <div style="width: 150pt">Paraformer-large</div> | <div style="width: 150pt">1.95</div>  | <div style="width: 150pt">1.68</div>     | 

### AISHELL-2

|           | dev_ios| test_android| test_ios|test_mic|
|:-------------------------------------------------:|:-------------------------------------:|:-------------------------------------:|:------------------------------------:|:------------------------------------:|
| <div style="width: 150pt">Espnet</div>            | <div style="width: 70pt">5.40</div>  |<div style="width: 70pt">6.10</div>  |<div style="width: 70pt">5.70</div>  |<div style="width: 70pt">6.10</div>  |
| <div style="width: 150pt">WeNet</div>             | <div style="width: 70pt">-</div>     |<div style="width: 70pt">-</div>     |<div style="width: 70pt">5.39</div>  |<div style="width: 70pt">-</div>    |
| <div style="width: 150pt">Paraformer-large</div>  | <div style="width: 70pt">2.80</div>  |<div style="width: 70pt">3.13</div>  |<div style="width: 70pt">2.85</div>  |<div style="width: 70pt">3.06</div>  |


### Wenetspeech

|           | dev| test_meeting| test_net|
|:-------------------------------------------------:|:-------------------------------------:|:-------------------------------------:|:------------------------------------:|
| <div style="width: 150pt">Espnet</div>            | <div style="width: 100pt">9.70</div>  |<div style="width: 100pt">15.90</div>  |<div style="width: 100pt">8.80</div>  |
| <div style="width: 150pt">WeNet</div>             | <div style="width: 100pt">8.60</div>  |<div style="width: 100pt">17.34</div>  |<div style="width: 100pt">9.26</div>  |
| <div style="width: 150pt">K2</div>                | <div style="width: 100pt">7.76</div>  |<div style="width: 100pt">13.41</div>  |<div style="width: 100pt">8.71</div>  |
| <div style="width: 150pt">Paraformer-large</div>  | <div style="width: 100pt">3.57</div>  |<div style="width: 100pt">6.97</div>   |<div style="width: 100pt">6.74</div>  |

### SpeechIO TIOBE

Paraformer-large模型结合Transformer-LM模型做shallow fusion，在公开评测项目SpeechIO TIOBE白盒测试场景上获得当前SOTA的效果，目前[Transformer-LM模型](https://modelscope.cn/models/damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch/summary)已在ModelScope上开源，以下展示SpeechIO TIOBE白盒测试场景without LM、with Transformer-LM的效果：

- Decode config w/o LM: 
  - Decode without LM
  - Beam size: 1
- Decode config w/ LM:
  - Decode with [Transformer-LM](https://modelscope.cn/models/damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch/summary)
  - Beam size: 10
  - LM weight: 0.15

| testset | w/o LM | w/ LM |
|:------------------:|:----:|:----:|
|<div style="width: 200pt">SPEECHIO_ASR_ZH00001</div>| <div style="width: 150pt">0.49</div> | <div style="width: 150pt">0.35</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH00002</div>| <div style="width: 150pt">3.23</div> | <div style="width: 150pt">2.86</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH00003</div>| <div style="width: 150pt">1.13</div> | <div style="width: 150pt">0.80</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH00004</div>| <div style="width: 150pt">1.33</div> | <div style="width: 150pt">1.10</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH00005</div>| <div style="width: 150pt">1.41</div> | <div style="width: 150pt">1.18</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH00006</div>| <div style="width: 150pt">5.25</div> | <div style="width: 150pt">4.85</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH00007</div>| <div style="width: 150pt">5.51</div> | <div style="width: 150pt">4.97</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH00008</div>| <div style="width: 150pt">3.69</div> | <div style="width: 150pt">3.18</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH00009</div>| <div style="width: 150pt">3.02</div> | <div style="width: 150pt">2.78</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH000010</div>| <div style="width: 150pt">3.35</div> | <div style="width: 150pt">2.99</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH000011</div>| <div style="width: 150pt">1.54</div> | <div style="width: 150pt">1.25</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH000012</div>| <div style="width: 150pt">2.06</div> | <div style="width: 150pt">1.68</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH000013</div>| <div style="width: 150pt">2.57</div> | <div style="width: 150pt">2.25</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH000014</div>| <div style="width: 150pt">3.86</div> | <div style="width: 150pt">3.08</div> |
|<div style="width: 200pt">SPEECHIO_ASR_ZH000015</div>| <div style="width: 150pt">3.34</div> | <div style="width: 150pt">2.67</div> |


## 使用方式以及适用范围

运行范围
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

使用方式
- 直接推理：可以直接对输入音频进行解码，输出目标文字。
- 微调：加载训练好的模型，采用私有或者开源数据进行模型训练。

使用范围与目标场景
- 适合与离线语音识别场景，如录音文件转写，配合GPU推理效果更加，推荐输入语音时长在20s以下，若想解码长音频，推荐使用[Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)，集成VAD、ASR、标点与时间戳功能，可直接对时长为数小时音频进行识别，并输出带标点文字与时间戳。


## 模型局限性以及可能的偏差

考虑到特征提取流程和工具以及训练工具差异，会对CER的数据带来一定的差异（<0.1%），推理GPU环境差异导致的RTF数值差异。



## 相关论文以及引用信息

```BibTeX
@inproceedings{gao2022paraformer,
  title={Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition},
  author={Gao, Zhifu and Zhang, Shiliang and McLoughlin, Ian and Yan, Zhijie},
  booktitle={INTERSPEECH},
  year={2022}
}
```
