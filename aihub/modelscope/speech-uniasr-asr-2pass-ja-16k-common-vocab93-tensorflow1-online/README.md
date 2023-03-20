# Highlights

- UniASR日语语音识别流式模型，可对近场、低噪、正常语速、朗读形式的日语音频进行语音识别：
  - ASR模型：[UniASR模型](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-id-16k-common-vocab93-tensorflow1-online/summary)，日语语音识别模型。
  - ITN模型：[日语ITN模型](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-ja/summary)，可用于语音识别结果进行拟文本正则化后处理。
  - VAD模型：[语音端点检查VAD模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)，可检测长语音片段中有效语音的起止时间点。
  
## Release Notes

- 2023年2月（2月17号发布）：[funasr-0.2.0](https://github.com/alibaba-damo-academy/FunASR/tree/main), modelscope-1.3.0
  - 功能完善：
    - 新增加模型导出功能，Modelscope中所有Paraformer模型与本地finetune模型，支持一键导出[onnx格式模型与torchscripts格式模型](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)，用于模型部署。
    - 新增加Paraformer模型[onnxruntime部署功能](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/onnxruntime/paraformer/rapid_paraformer)，无须安装Modelscope与FunASR，即可部署，cpu实测，onnxruntime推理速度提升近3倍(rtf: 0.110->0.038)。
    - 新增加[grpc服务功能](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/grpc)，支持对Modelscope推理pipeline进行服务部署，也支持对onnxruntime进行服务部署。
    - 优化[Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)时间戳，对badcase时间戳预测准确率有较大幅度提升。
    - 新增加任意VAD模型、ASR模型与标点模型自由组合功能，可以自由组合Modelscope中任意模型以及本地finetune后的模型进行推理。
    - 新增加采样率自适应功能，任意输入采样率音频会自动匹配到模型采样率；新增加多种语音格式支持，如，mp3、flac、ogg、opus等。

  - 上线新模型：
    - [Paraformer-large热词模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary)，可实现热词定制化，基于提供的热词列表，对热词进行激励增强，提升模型对热词的召回。
    - [MFCCA多通道多说话人识别模型](https://www.modelscope.cn/models/yufan6/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary)，与西工大音频语音与语言处理研究组合作论文，一种基于多帧跨通道注意力机制的多通道语音识别模型。
    - [8k语音端点检测VAD模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-8k-common/summary)，可用于检测长语音片段中有效语音的起止时间点，支持流式输入，最小支持10ms语音输入流。
    - UniASR流式离线一体化模型:
    [16k UniASR闽南语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-minnan-16k-common-vocab3825/summary)、
    [16k UniASR法语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fr-16k-common-vocab3472-tensorflow1-online/summary)、
    [16k UniASR德语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-de-16k-common-vocab3690-tensorflow1-online/summary)、
    [16k UniASR越南语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online/summary)、
    [16k UniASR波斯语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fa-16k-common-vocab1257-pytorch-online/summary)。
    - [基于Data2vec结构无监督预训练Paraformer模型](https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-paraformer-zh-cn-aishell2-16k/summary)，采用Data2vec无监督预训练初值模型，在AISHELL-1数据中finetune Paraformer模型。

- 2023年1月（1月16号发布）：[funasr-0.1.6](https://github.com/alibaba-damo-academy/FunASR/tree/main), modelscope-1.2.0
  - 上线新模型：
    - [Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)，集成VAD、ASR、标点与时间戳功能，可直接对时长为数小时音频进行识别，并输出带标点文字与时间戳。
    - [中文无监督预训练Data2vec模型](https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch/summary)，采用Data2vec结构，基于AISHELL-2数据的中文无监督预训练模型，支持ASR或者下游任务微调模型。
    - [16k语音端点检测VAD模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)，可用于检测长语音片段中有效语音的起止时间点。
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

UniASR模型是一种2遍刷新模型（Two pass）端到端语音识别模型。日益丰富的业务需求，不仅要求识别效果精度高，而且要求能够实时地进行语音识别。一方面，离线语音识别系统具有较高的识别准确率，但其无法实时的返回解码文字结果，并且，在处理长语音时，容易发生解码重复的问题，以及高并发解码超时的问题等；另一方面，流式系统能够低延时的实时进行语音识别，但由于缺少下文信息，流式语音识别系统的准确率不如离线系统，在流式业务场景中，为了更好的折中实时性与准确率，往往采用多个不同时延的模型系统。为了满足差异化业务场景对计算复杂度、实时性和准确率的要求，常用的做法是维护多种语音识别系统，例如，CTC系统、E2E离线系统、SCAMA流式系统等。在不同的业务场景使用不同的模型和系统，不仅会增加模型生产成本和迭代周期，而且会增加引擎以及服务部署的维护成本。因此，我们设计了离线流式一体化语音识别系统——UniASR。UniASR同时具有高精度和低延时的特点，不仅能够实时输出语音识别结果，而且能够在说话句尾用高精度的解码结果修正输出，与此同时，UniASR采用动态延时训练的方式，替代了之前维护多套延时流式系统的做法。通过设计UniASR语音识别系统，我们将之前多套语音识别系统架构统一为一套系统架构，一个模型满足所有业务场景，显著的降低了模型生产和维护成本。
其模型结构如下图所示：

<div align=center>
<img src="fig/UniASR.png" width="850" height="270"/>
</div>

UniASR模型结构如上图所示，包含离线语音识别部分和流式语音识别部分。其中，离线与流式部分通过共享一个动态编码器（Encoder）结构来降低计算量。流式语音识别部分是由动态时延 Encoder 与流式解码器（Decoder）构成。动态时延 Encoder 采用时延受限有句记忆单元的自注意力（LC-SAN-M）结构；流式 Decoder 采用动态 SCAMA 结构。离线语音识别部分包含了降采样层（Stride Conv）、Big-Chunk Encoder、文本Encoder与SCAMA Decoder。为了降低刷新输出结果的尾点延时，离线识别部分采用大Chunk 流式结构。其中，Stride Conv结构是为了降低计算量。文本 Encoder 增加了离线识别的语义信息。为了让模型能够具有不同延时下进行语音识别的能力，我们创新性地设计了动态时延训练机制，使得模型能够同时满足不同业务场景对延时和准确率的要求。
根据业务场景特征，我们将语音识别需求大致分为3类： 

    低延迟实时听写：如电话客服，IOT语音交互等，该场景对于尾点延迟非常敏感，通常需要用户说完以后立马可以得到识别结果。  
    流式实时听写：如会议实时字幕，语音输入法等，该场景不仅要求能够实时返回语音识别结果，以便实时显示到屏幕上，而且还需要能够在说话句尾用高精度识别结果刷新输出。  
    离线文件转写：如音频转写，视频字幕生成等，该场景不对实时性有要求，要求在高识别准确率情况下，尽可能快的转录文字。  

为了同时满足上面3种业务场景需求，我们将模型分成3种解码模式，分别对应为：  

    fast 模式：只有一遍解码，采用低延时实时出字模式；  
    normal 模式：2遍解码，第一遍低延时实时出字上屏，第二遍间隔3～6s（可配置）对解码结果进行刷新；  
    offline 模式：只有一遍解码，采用高精度离线模式；  

在模型部署阶段，通过发包指定该次语音识别服务的场景模式和延时配置。这样，通过UniASR系统，我们统一了离线流式语音识别系统架构，提高模型识别效果的同时，不仅降低了模型生产成本和迭代周期，还降低了引擎以及服务部署维护成本。目前我们提供的语音识别服务基本都是基于UniASR。

## 如何训练自己的UniASR模型

本项目提供的UniASR模型是通用领域识别模型，开发者可以基于此模型进一步利用ModelScope的微调功能或者本项目对应的github代码仓库[FunASR](https://github.com/alibaba-damo-academy/FunASR)进一步进行模型的领域定制化。

### 在Notebook中开发

对于有开发需求的使用者，特别推荐您使用Notebook进行离线处理。先登录ModelScope账号，点击模型页面右上角的“在Notebook中打开”按钮出现对话框，首次使用会提示您关联阿里云账号，按提示操作即可。关联账号后可进入选择启动实例界面，选择计算资源，建立实例，待实例创建完成后进入开发环境，进行调用。

#### 基于ModelScope进行推理

- 推理支持音频格式如下：
  - wav文件路径，例如：data/test/audios/asr_example.wav
  - wav文件url，例如：https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_ja.wav
  - wav二进制数据，格式bytes，例如：用户直接从文件里读出bytes数据或者是麦克风录出bytes数据。
  - 已解析的audio音频，例如：audio, rate = soundfile.read("asr_example_ja.wav")，类型为numpy.ndarray或者torch.Tensor。
  - wav.scp文件，需符合如下要求：

```sh
cat wav.scp
asr_example1  data/test/audios/asr_example1.wav
asr_example2  data/test/audios/asr_example2.wav
...
```

- 若输入格式wav文件url，api调用方式可参考如下范例： 
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online')

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_ja.wav')
print(rec_result)
```

- 切换fast、normal、offline解码模式(默认normal模式)
```python
rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_ja.wav', param_dict={"decoding_model": "normal"})
```

- 若输入格式为文件wav.scp(注：文件名需要以.scp结尾)，可添加 output_dir 参数将识别结果写入文件中，参考示例如下：

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online',
    output_dir='./output_dir')

inference_pipeline("wav.scp")
```
识别结果输出路径结构如下：

```sh
tree output_dir/
output_dir/
└── 1best_recog
    ├── rtf
    ├── score
    ├── text
    └── time_stamp

1 directory, 4 files
```
rtf：计算过程耗时统计

score：识别路径得分

text：语音识别结果文件

time_stamp：时间戳结果文件

- 若输入音频为已解析的audio音频，api调用方式可参考如下范例：
```python
import soundfile
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online')

waveform, sample_rate = torchaudio.load("asr_example_ja.wav")
rec_result = inference_pipeline(audio_in=waveform)
print(rec_result)
```

#### 基于ModelScope进行微调

- 基于ModelScope上数据集进行微调：

以[日语语音识别Common Voice数据集](https://modelscope.cn/datasets/modelscope/speech_asr_commonvoice_ja_trainsets/summary)数据集为例，完整数据集已经上传ModelScope，可通过数据集英文名（speech_asr_commonvoice_ja_trainsets）搜索：


```python
import os
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer

exp_dir = './exp'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# dataset split ['train', 'validation']
ds_dict = MsDataset.load(dataset_name='speech_asr_commonvoice_ja_trainsets', namespace='modelscope')

kwargs = dict(
    model='damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online',
    data_dir=ds_dict,
    work_dir=exp_dir,
    distributed=False)

trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
trainer.train()
```


若想使用多卡进行微调训练，可将distributed参数改为True，参考如下：

```python
kwargs = dict(
    model='damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online',
    data_dir=ds_dict,
    work_dir=exp_dir,
    distributed=True)
```

- 基于私有数据集进行微调:

私有数据集格式按如下准备：
```sh
tree ./example_data/
./example_data/
├── dev
│   ├── text
│   └── wav.scp
└── train
    ├── text
    └── wav.scp
2 directories, 4 files
```

其中，text文件中存放音频标注，wav.scp文件中存放wav音频绝对路径，样例如下：

```sh
cat ./example_data/text
common_voice_ja_25361422  halo dunia
common_voice_ja_25377216  sudah makan sudah sholat

cat ./example_data/wav.scp
common_voice_ja_25361422 /mnt/data/train/common_voice_ja_25361422.wav
common_voice_ja_25377216 /mnt/data/train/common_voice_ja_25377216.wav
```

安装[FunASR](https://github.com/alibaba-damo-academy/FunASR)框架，安装命令如下：

```
git clone https://github.com/alibaba/FunASR.git
cd FunASR
pip install --editable ./
```

训练私有数据代码范例如下：

```python
import os
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from funasr.datasets.ms_dataset import MsDataset

def modelscope_finetune(params):
    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"], exist_ok=True)
    # dataset split ["train", "validation"]
    ds_dict = MsDataset.load(params["data_dir"])
    kwargs = dict(
        model=params["model"],
        data_dir=ds_dict,
        dataset_type=params["dataset_type"],
        work_dir=params["output_dir"],
        batch_bins=params["batch_bins"],
        max_epoch=params["max_epoch"],
        lr=params["lr"])
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()

if __name__ == '__main__':
    params = {}
    params["output_dir"] = "./checkpoint"              # 保存路径
    params["data_dir"] = "./example_data/"             # 数据路径
    params["batch_bins"] = 16000                       # batch size
    params["dataset_type"] = "small"                   # 小数据量设置small，若数据量大于1000小时，请使用large
    params["max_epoch"] = 50                           # 最大训练轮数
    params["lr"] = 0.00005                             # 设置学习率
    params["model"] = "damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online"
    modelscope_finetune(params)
```

若使用多卡进行训练，可将上述代码保存为py文件（如finetune.py）后，执行如下命令：

```sh
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 finetune.py > 1234.txt 2>&1
```

### 在本地机器中开发

#### 基于ModelScope进行微调和推理

支持基于ModelScope上数据集及私有数据集进行定制微调和推理，使用方式同Notebook中开发。

#### 基于FunASR进行微调和推理

FunASR框架支持魔搭社区开源的工业级的语音识别模型的training & finetuning，使得研究人员和开发者可以更加便捷的进行语音识别模型的研究和生产，目前已在github开源：https://github.com/alibaba-damo-academy/FunASR

#### FunASR框架安装

- 安装FunASR和ModelScope

```sh
pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/alibaba/FunASR.git
cd FunASR
pip install --editable ./
```

#### 基于FunASR进行推理

接下来会以私有数据集为例，介绍如何在FunASR框架中使用UniASR进行推理以及微调。

```sh
cd egs_modelscope/asr/uniasr/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online/
python infer.py
```

#### 基于FunASR进行微调
```sh
cd egs_modelscope/asr/uniasr/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online/
python finetune.py
```

data_dir参数为私有数据集路径，其数据格式可参考基于ModelScope微调中的数据格式要求。若想使用多卡进行微调训练，可添加参数，如下所示：
```sh
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 finetune.py > 1234.txt 2>&1
```

## 使用方式以及适用范围

运行范围
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

使用方式
- 直接推理：可以直接对输入音频进行解码，输出目标文字。
- 微调：加载训练好的模型，采用私有或者开源数据进行模型训练。

使用范围与目标场景
- 适合于近场、低噪、正常语速、朗读形式语音识别场景，如录音文件转写，建议配合VAD模型处理输入音频。

### 模型局限性以及可能的偏差

考虑到特征提取流程和工具以及训练工具差异，会对CER的数据带来一定的差异（<0.1%），推理GPU环境差异导致的RTF数值差异。

### 预处理

可以直接采用原始音频作为输入进行训练，也可以先对音频进行预处理，提取FBank特征，再进行模型训练，加快训练速度。

### 后处理

[FunTextProcessing](https://github.com/alibaba-damo-academy/FunASR/tree/main/fun_text_processing)支持魔搭社区开源的工业级的语音识别的后处理ITN、TN功能，使得研究人员和开发者可以更加便捷的对语音识别模型结果进行后处理的研究和生产，目前已在github开源：[https://github.com/alibaba-damo-academy/FunASR/tree/main/fun_text_processing](https://github.com/alibaba-damo-academy/FunASR/tree/main/fun_text_processing)

### 版本历史
- v1.0.2 Pytorch Japanese Characters Model Version
- v1.0.1 Pytorch Kanataka Model Version
- v1.0.0 Tensorflow Kanataka Model Version

### 相关论文以及引用信息

```BibTeX
@inproceedings{gao2020universal,
  title={Universal ASR: Unifying Streaming and Non-Streaming ASR Using a Single Encoder-Decoder Model},
  author={Gao, Zhifu and Zhang, Shiliang and Lei, Ming and McLoughlin, Ian},
  booktitle={arXiv preprint arXiv:2010.14099},
  year={2010}
}
```
