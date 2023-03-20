
# Highlights
说话人确认和声纹提取模型，训练数据集
- 多领域数据 CN-Celeb 1&2
- 会议场景 AliMeeting

支持功能：
- 提取一段语音的说话人嵌入码 speaker embedding
- 说话人确认：判断两段语音是否为同一说话人

# Release Note

- 2023年2月（2月17号发布）：[funasr-0.2.0](https://github.com/alibaba-damo-academy/FunASR/tree/main), modelscope-1.3.0
  - 功能完善：
    - 新增加模型导出功能，Modelscope中所有Paraformer模型与本地finetune模型，支持一键导出[onnx格式模型与torchscripts格式模型](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)，用于模型部署。
    - 新增加Paraformer模型[onnxruntime部署功能](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/onnxruntime/paraformer/rapid_paraformer)，无须安装Modelscope与FunASR，即可部署，cpu实测，onnxruntime推理速度提升近3倍(rtf: 0.110->0.038)。
    - 新增加[grpc服务功能](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/grpc)，支持对Modelscope推理pipeline进行服务部署，也支持对onnxruntime进行服务部署。
    - 优化[Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)时间戳，对badcase时间戳预测准确率有较大幅度提升，平均首尾时间戳偏移74.7ms，[详见论文](https://arxiv.org/abs/2301.12343)。
    - 新增加任意VAD模型、ASR模型与标点模型自由组合功能，可以自由组合Modelscope中任意模型以及本地finetune后的模型进行推理。
    - 新增加采样率自适应功能，任意输入采样率音频会自动匹配到模型采样率；新增加多种语音格式支持，如，mp3、flac、ogg、opus等。

  - 上线新模型：
    - [Paraformer-large热词模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary)，可实现热词定制化，基于提供的热词列表，对热词进行激励增强，提升模型对热词的召回。
    - [MFCCA多通道多说话人识别模型](https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary)，与西工大音频语音与语言处理研究组合作论文，一种基于多帧跨通道注意力机制的多通道语音识别模型。
    - [8k语音端点检测VAD模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-8k-common/summary)，可用于检测长语音片段中有效语音的起止时间点，支持流式输入，最小支持10ms语音输入流。
    - UniASR流式离线一体化模型: 
    [16k UniASR闽南语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-minnan-16k-common-vocab3825/summary)、
    [16k UniASR法语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fr-16k-common-vocab3472-tensorflow1-online/summary)、
    [16k UniASR德语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-de-16k-common-vocab3690-tensorflow1-online/summary)、
    [16k UniASR越南语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online/summary)、
    [16k UniASR波斯语](https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-fa-16k-common-vocab1257-pytorch-online/summary)。
    - [基于Data2vec结构无监督预训练Paraformer模型](https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-paraformer-zh-cn-aishell2-16k/summary)，采用Data2vec无监督预训练初值模型，在AISHELL-1数据中finetune Paraformer模型。

- 2023年1月（预计1月16号发布）：funasr-0.1.6, modelscope-1.1.4
  - 模型功能完善：
    - Modelscope模型推理pipeline，新增加多种输入音频方式，如wav.scp、音频bytes、音频采样点、MP3格式、录音笔格式等。
    - [Paraformer-large模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)，新增加基于ModelScope微调定制模型，新增加batch级解码，加快推理速度。
    - [AISHELL-1学术集Paraformer模型](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)，
    [AISHELL-1学术集ParaformerBert模型](https://modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)，
    [AISHELL-1学术集Conformer模型](https://modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary)、
    [AISHELL-2学术集Paraformer模型](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)，
    [AISHELL-2学术集ParaformerBert模型](https://www.modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)、
    [AISHELL-2学术集Conformer模型](https://www.modelscope.cn/models/damo/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary)，
    新增加基于ModelScope微调定制模型，其中，Paraformer与ParaformerBert模型新增加batch级解码，加快推理速度。
  - 上线新模型：
    - [说话人确认模型](https://www.modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary) ，可用于说话人确认，也可以用来做说话人特征提取。
    - [Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)，集成VAD、ASR、标点与时间戳功能，可直接对时长为数小时音频进行识别，并输出带标点文字与时间戳。
    - [中文无监督预训练Data2vec模型](https://www.modelscope.cn/models/damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch/summary)，采用Data2vec结构，基于AISHELL-2数据的中文无监督预训练模型，支持ASR或者下游任务微调模型。
    - [语音端点检查VAD模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)，可用于检测长语音片段中有效语音的起止时间点。
    - [中文标点预测通用模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)，可用于语音识别模型输出文本的标点预测。
    - [8K UniASR流式模型](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online/summary)，[8K UniASR模型](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-offline/summary)，一种流式与离线一体化语音识别模型，进行流式语音识别的同时，能够以较低延时输出离线识别结果来纠正预测文本。
    - Paraformer-large基于[AISHELL-1微调模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell1-vocab8404-pytorch/summary)、[AISHELL-2微调模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell2-vocab8404-pytorch/summary)，将Paraformer-large模型分别基于AISHELL-1与AISHELL-2数据微调。
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
  - Paraformer-large非自回归模型上线，多个公开数据集上取得SOTA效果：
    - 支持基于ModelScope推理。
    - 支持基于[FunASR框架开源](https://github.com/alibaba-damo-academy/FunASR)微调和推理。

# 项目介绍

ResNet34是说话人确认（Speaker Verification）中常用的模型结构。
模型包括帧级说话人特征提取主干网络ResNet34、全局统计池化global statistic pooling以及多个全连接层fully-connected layer。
该模型是达摩院语音团队在常用开源数据集CN-Celeb 1&2和会议场景数据集AliMeeting上预训练得到的，
可以用于通用任务的说话人嵌入码（speaker embedding）提取，或进行说话人确认任务。
该模型还可以用于说话人日志（speaker diarization）任务，并取得了良好的识别效果。
具体可以参考我们发表于EMNLP 2022上的<a href="https://arxiv.org/abs/2211.10243">论文</a>和
<a href="https://mp.weixin.qq.com/s/iU09MDjcFTaIJXIjc9isIA">论文解读</a>。

<p align="center">
<img src="fig/struct.png" alt="Speaker Verification模型结构"  width="1102" />
</p>

# 如何使用模型

## 在线快速体验
在页面右侧，可以在“在线体验”栏内看到我们预先准备好的示例音频，点击播放按钮可以试听，点击“执行测试”按钮，会在下方“测试结果”栏中显示两个语音说话人之间的相似程度（0到1之间）。如果您想要测试自己的音频，可点击“更换音频”按钮，选择上传或录制一段音频，完成后点击执行测试，相似度将会在测试结果栏中显示。

## 在Notebook中推理
对于灵活调用有需求的开发者，我们推荐您使用Notebook进行处理。首先登录ModelScope账号，点击模型页面右上角的“在Notebook中打开”按钮出现对话框，首次使用会提示您关联阿里云账号，按提示操作即可。关联账号后可进入选择启动实例界面，选择计算资源，建立实例，待实例创建完成后进入开发环境，输入api调用实例。

- 使用本模型提取说话人嵌入码（speaker embedding）：
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np

inference_sv_pipline = pipeline(
    task=Tasks.speaker_verification,
    model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
)

# 对于单个url我们使用"spk_embedding"作为key
spk_embedding = inference_sv_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav')["spk_embedding"]
```
- 除了url表示的网络wav文件，还可使用本地磁盘上的wav文件：
```python
spk_embedding = inference_sv_pipline(audio_in='sv_example_enroll.wav')["spk_embedding"]
```
- 以及已经读取到内存中的numpy数组或者pytorch张量（Tensor）
```python
import soundfile
wav = soundfile.read('sv_example_enroll.wav', dtype="int16")[0]
# 对于内存中的数组或者张量我们使用"spk_embedding"作为key
spk_embedding = inference_sv_pipline(audio_in=wav)["spk_embedding"]
```
- speaker embedding的一个主要应用是评估两个说话人的相似程度：

```python
enroll = inference_sv_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav')["spk_embedding"]

same = inference_sv_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_same.wav')["spk_embedding"]

import numpy as np
# 对相同的说话人计算余弦相似度
sv_threshold=0.9465
same_cos=np.sum(enroll*same)/(np.linalg.norm(enroll)*np.linalg.norm(same))
same_cos=max(same_cos - sv_threshold, 0.0) / (1.0 - sv_threshold) * 100.0
print(same_cos)
```

- 为了方便使用，本模型也支持直接进行说话人确认（speaker verification）：
```python
# 两个语音为相同说话人
rec_result = inference_sv_pipline(audio_in=('https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav','https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_same.wav'))
print(rec_result["scores"][0])

# 两个语音为不同说话人
rec_result = inference_sv_pipline(audio_in=('https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav','https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_different.wav'))

print(rec_result["scores"][0])
```

## 在本地机器中推理

如果您有本地推理或定制模型的需求，可以前往下载FunASR语音处理框架，不仅涵盖语音识别、端点检测和说话人确认等多种模型，还支持ModelScope开源模型的推理，使研究人员和开发者可以更加便捷的进行模型研究和生产，目前已在github开源：<a href="https://github.com/alibaba-damo-academy/FunASR">https://github.com/alibaba-damo-academy/FunASR</a>

### FunASR框架安装

- 安装FunASR和ModelScope

```sh
# 安装 Pytorch GPU (version >= 1.7.0):
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch  
# 对于其他版本，请参考 https://pytorch.org/get-started/locally

# 安装 ModelScope包:
pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

# 下载项目代码:
git clone https://github.com/alibaba/FunASR.git

# 安装 FunASR:
pip install --editable ./
```

### 基于ModelScope进行推理

- 在上面的安装完成后，就可以在使用ModelScope进行推理了，可运行如下命令提取speaker embedding：
```sh
cd egs_modelscope/speaker_verification/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch
python infer.py 
```
- 或测试说话人确认功能：
```sh
python infer_sv.py
```

### 基于ModelScope在CN-Celeb测试集上进行性能评估
接下以CN-Celeb数据集为例，介绍如何使用ModelScope对模型的EER、minDCF等性能指标进行评估，从Openslr上下载完整的CN-Celeb 1&2数据集：<a href="http://openslr.org/82/">http://openslr.org/82/</a>
```sh
# 进入工作目录
cd egs_modelscope/speaker_verification/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch

# 获取评估用的数据
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/cnceleb-eval.tar.gz
tar zxf cnceleb-eval.tar.gz

# 进行评估
python ./eval_eer.py
```

### 基于ModelScope进行模型微调

训练和微调功能正在开发中，敬请期待。

# Benchmark

## 训练配置
- Feature info: using 80 dims fbank, no cmvn, speed perturb(0.9, 1.0, 1.1)
- Train info: lr 1e-4, batch_size 64, 1 gpu(Tesla V100), acc_grad 1, 300000 steps, clip_gradient_norm 3.0, weight_l2_regularizer 0.01
- Loss info: additive angular margin softmax, feature_scaling_factor=8, margin 0.25
- Model info: ResNet34, global statistics pooling, Dense
- Train config: sv.yaml
- Model size: 5.60 M parameters

## 实验结果 (EER & minDCF)
- Test set: Alimeeting-test, CN-Celeb-eval-speech

|       testset         | EER(%)  |  minDCF | Threshold |
|:---------------------:|:-------:|:-------:| :--------:| 
|    Alimeeting-test    |  1.45   | 0.0849  | 0.9666    |
|  CN-Celeb-eval-speech |  9.00   | 0.2936  | 0.9465    |

## 使用方式以及适用范围

运行范围
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

使用方式
- 直接推理：可以提取语音的说话人 embedding，或者判断两句话的说话人是否相同。
- 微调：正在开发中。

使用范围与目标场景
- 适合于学术研究，在CN-Celeb、AliMeeting等数据集上进行说话人日志、识别等任务。

## 模型局限性以及可能的偏差

- 特征提取流程和工具差异，会对EER的数值带来一定的差异（<0.1%）。
- 语句的过长、过短或静音过多会对性能产生一定影响
