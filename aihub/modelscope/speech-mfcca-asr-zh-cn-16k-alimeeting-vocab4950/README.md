
## 模型描述

多说话人语音识别(Multi-talker ASR)的目标是识别包含多个说话人的语音，希望能够正确识别极具挑战的说话人重叠（speaker overlap）的语音。近年来，随着深度学习的发展，许多端到端多说话人ASR的方法出现，并在多说话人模拟数据集(如LibriCSS)上取得了良好的效果。然而，包括会议在内的真实场景中包含了更多挑战，如说话人重叠率较高的多人讨论、自由对话风格的语音、说话人数量未知、远场语音信号衰减、噪声和混响干扰等。其中如何利用好麦克风阵列（microphone array）录制的多通道音频，是一直以来的研究重心。最近，跨通道注意力机制在多方会议场景中显示出了优越的效果，能够高效地利用麦克风阵列的多通道信号帮助提升语音识别的性能。目前主要有两类方法，分别为帧级和通道级的跨通道注意力机制。前者注重学习不同通道序列之间的全局相关性，后者注重在每个时间步中对通道信息进行细粒度地建模。


近期，西工大音频语音与语言处理研究组（ASLP@NPU）和阿里巴巴达摩院高校AIR合作论文“MFCCA:Multi-frame cross-channel attention for multi-speaker ASR in multi-party meeting scenario”被语音旗舰会议SLT 2022接收。该论文考虑到麦克风阵列不同麦克风接收信号的差异，提出了一种多帧跨通道注意力机制（MFCCA），该方法对相邻帧之间的跨通道信息进行建模，以利用帧级和通道级信息的互补性。此外，该论文还提出了一种多层卷积模块以融合多通道输出和一种通道掩码策略以解决训练和推理之间的音频通道数量不匹配的问题。在ICASSP2022 M2MeT竞赛上发布的真实会议场景语料库AliMeeting上进行了相关实验，该多通道模型在Eval和Test集上比单通道模型CER分别相对降低了39.9%和37.0%。此外，在同等的模型参数和训练数据下，本文提出的模型获得的识别性能超越竞赛期间最佳结果，在AliMeeting上实现了目前最新的SOTA性能。


<div align=center>
<img src="fig/mfcca_attention.png"/>
</div>



帧级跨通道注意力机制(Frame-Level Cross-Channel Attention，FLCCA)如上图（b）所示，其将帧级的多通道信号作为输入，然后学习不同通道序列之间的全局相关性。简单来说，就是将每个通道的高维表示(query)与一组通道平均高维表示对(key-value)映射到输出。
通道级跨通道注意力机制(Channel-Level Cross-Channel Attention，CLCCA)如上图（c）所示，其是在通道的维度上进行计算，也就是对于每个时间步中的通道信息进行注意力机制的计算，起到了一个类似于波束形成的作用。
我们认为帧级跨通道注意力机制和通道级跨通道注意力机制获取时序和空间信息时是可以互补的，提出了一种多帧跨通道注意力机制(Multi-Frame Cross-Channel Attention，MFCCA)，如上图（d）所示，其更多地关注相邻帧之间的通道上下文，以建模帧级和通道级的相关性。

更详细的细节见：
- 论文： [MFCCA:Multi-Frame Cross-Channel attention for multi-speaker ASR in Multi-party meeting scenario](https://arxiv.org/abs/2210.05265)
- 论文解读：[论文推介：MFCCA--基于多帧跨通道注意力机制的多说话人语音识别](https://mp.weixin.qq.com/s/23QlNzTtpzjSSlpIlUAf2A)


## 如何使用与训练自己的模型

本项目提供的预训练模型正如论文所述，是基于AliMeeting、AISHELL-4和700h模拟重叠音频共917h的训练数据训练而成的多通道多说话人识别模型，开发者可以基于此模型进一步利用ModelScope的微调功能或者本项目对应的Github代码仓库[FunASR](https://github.com/alibaba-damo-academy/FunASR)进一步进行模型的领域定制化。


### 在Notebook中开发

对于有开发需求的使用者，特别推荐您使用Notebook进行离线处理。先登录ModelScope账号，点击模型页面右上角的“在Notebook中打开”按钮出现对话框，首次使用会提示您关联阿里云账号，按提示操作即可。关联账号后可进入选择启动实例界面，选择计算资源，建立实例，待实例创建完成后进入开发环境，进行调用。

#### 基于ModelScope进行推理

- 推理支持音频格式如下：
  - wav文件路径，例如：data/test/audios/asr_example.wav
  - pcm文件路径，例如：data/test/audios/asr_example.pcm
  - wav文件url，例如：https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav
  - wav二进制数据，格式bytes，例如：用户直接从文件里读出bytes数据或者是麦克风录出bytes数据。
  - 已解析的audio音频，例如：audio, rate = soundfile.read("asr_example_zh.wav")，类型为numpy.ndarray或者torch.Tensor。
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
    model='NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950',
    model_revision='v3.0.0')

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_vad_punc_example.wav')
print(rec_result)
```

- 输入音频为pcm格式，调用api时需要传入音频采样率参数audio_fs，例如：

```python
rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_vad_punc_example.pcm', audio_fs=16000)
```

- 输入音频为wav格式，api调用方式可参考如下范例:

```python
rec_result = inference_pipeline(audio_in='asr_example_mc.wav')
```

- 若输入格式为文件wav.scp(注：文件名需要以.scp结尾)，可添加 output_dir 参数将识别结果写入文件中，api调用方式可参考如下范例:

```python
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950',
    model_revision='v3.0.0',
    output_dir='./output_dir')

inference_pipeline(audio_in="wav.scp")
```
识别结果输出路径结构如下：

```sh
tree output_dir/
output_dir/
└── 1best_recog
    ├── ref_text_nosep
    ├── score
    ├── text
    ├── token
    ├── token_nosep
    ├── text.sp.cer
    └── text.nosp.cer

1 directory, 7 files
```
ref_text_nosep：去掉分隔符的抄本

score：识别路径得分

text：语音识别结果文件

token：语音识别结果文件，中间用空格隔开

token_nosep：语音识别结果文件，中间用空格隔开，去掉分隔符

text.sp.cer：保留分隔符计算的CER结果

text.nosp.cer：去掉分隔符计算的CER结果（ICASSP2022 M2MeT挑战赛结果不计算分隔符）

- 若输入音频为已解析的audio音频，api调用方式可参考如下范例：

```python
import soundfile

waveform, sample_rate = soundfile.read("asr_example_mc.wav")
rec_result = inference_pipeline(audio_in=waveform)
```


#### 基于ModelScope进行微调

- 基于AliMeeting数据集进行微调：

[AliMeeting](https://www.openslr.org/119)数据集可以在openslr上下载。


[数据处理脚本]（https://github.com/yufan-aslp/AliMeeting）可以参考ICASSP2022 M2MeT挑战赛的Baseline，如果要跑多通道的数据的话，可以把baseline脚本生成的wav.scp文件中的-c 1改成-c 8，从而生成多通道数据。后续我们也会在Funasr上重新整理数据处理脚本。


AliMeeting数据集格式按如下准备：
```sh
tree ./example_data/
./example_data/
├── validation
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
Train-far-mc-R0003_M0046_MS002-R0003_M0046_F_SPK0093-0011198-0011987 啊 是 src 其 实 这 种 的 话 完 全 可 以 取 决 于 他 结 婚 的 那 个 场 地 比 如 说 他 在 室 内 我 们 可 以 开 空 调 的 对 不 对 像 酒 店 什 么 的 src 啊 还 是 
Train-far-mc-R0003_M0046_MS002-R0003_M0046_F_SPK0093-0013847-0014799 嗯 src 嗯 src 那 如 果 是 秋 天 举 办 的 话 可 能 就 是 正 逢 那 些 秋 叶 什 么 的 都 是 黄 色 的 就 是 金 色 系 的 这 个 也 也 要 看 客 户 他 们 喜 欢 什 么 色 系 的 他 们 有 说 吗 老 师

cat ./example_data/wav.scp
Train-far-mc-R0003_M0046_MS002-R0003_M0046_F_SPK0093-0011198-0011987 /home/work_nfs5_ssd/fyu/workspace/FunASR/egs_modelscope/alimeeting/mfcca/dump/raw/org/Train_Ali_far_multichannel/data/format.1/Train-far-mc-R0003_M0046_MS002-R0003_M0046_F_SPK0093-0011198-0011987.wav
Train-far-mc-R0003_M0046_MS002-R0003_M0046_F_SPK0093-0013847-0014799 /home/work_nfs5_ssd/fyu/workspace/FunASR/egs_modelscope/alimeeting/mfcca/dump/raw/org/Train_Ali_far_multichannel/data/format.1/Train-far-mc-R0003_M0046_MS002-R0003_M0046_F_SPK0093-0013847-0014799.wav

```

安装[FunASR](https://github.com/alibaba-damo-academy/FunASR)框架，安装命令如下：

```
git clone https://github.com/alibaba/FunASR.git
cd FunASR
pip install --editable ./
```

代码范例如下：

```python
import os
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from funasr.datasets.ms_dataset import MsDataset
from funasr.utils.modelscope_param import modelscope_args

def modelscope_finetune(params):
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir, exist_ok=True)
    # dataset split ["train", "validation"]
    ds_dict = MsDataset.load(params.data_path)
    kwargs = dict(
        model=params.model,
        model_revision=params.model_revision,
        data_dir=ds_dict,
        dataset_type=params.dataset_type,
        work_dir=params.output_dir,
        batch_bins=params.batch_bins,
        max_epoch=params.max_epoch,
        lr=params.lr)
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    
    params = modelscope_args(model="NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950")
    params.output_dir = "./checkpoint"              # m模型保存路径
    params.data_path = "./example_data/"            # 数据路径
    params.dataset_type = "small"                   # 小数据量设置small，若数据量大于1000小时，请使用large
    params.batch_bins = 1000                       # batch size，如果dataset_type="small"，batch_bins单位为fbank特征帧数，如果dataset_type="large"，batch_bins单位为毫秒，
    params.max_epoch = 10                           # 最大训练轮数
    params.lr = 0.0001                             # 设置学习率
    params.model_revision = 'v3.0.0'
    modelscope_finetune(params)
```

可将上述代码保存为py文件（如finetune.py），在notebook里面执行，若使用多卡进行训练，如下命令：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4  finetune.py > 1234.txt 2>&1
```

如果报P2P的错，可以加上NCCL_IGNORE_DISABLED_P2P=1，如下命令：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_IGNORE_DISABLED_P2P=1  python -m torch.distributed.launch --nproc_per_node 4  finetune.py > 1234.txt 2>&1
```


### 在本地机器中开发

#### 基于ModelScope进行微调和推理

支持基于ModelScope上数据集及私有数据集进行定制微调和推理，使用方式同Notebook中开发。

#### 基于FunASR进行微调和推理

FunASR框架支持魔搭社区开源的工业级的语音识别模型的training & finetuning，使得研究人员和开发者可以更加便捷的进行语音识别模型的研究和生产，目前已在Github开源：https://github.com/alibaba-damo-academy/FunASR 。若在使用过程中遇到任何问题，欢迎联系我们：[联系方式](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/images/dingding.jpg)

#### FunASR框架安装

- 安装FunASR和ModelScope

```sh
pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/alibaba/FunASR.git
cd FunASR
pip install --editable ./
```


#### 基于FunASR进行推理

接下来会以AliMeeting数据集为例，介绍如何在FunASR框架中使用MFCCA进行推理以及微调。

```sh
cd egs_modelscope/asr/mfcca/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950
python infer.py
```

#### 基于FunASR进行微调
```sh
cd egs_modelscope/asr/mfcca/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950
python finetune.py
```

若修改输出路径、数据路径、采样率、batch_size等配置，可参照在Notebook开发中私有数据微调部分的代码，修改finetune.py文件中配置，例如：
```python
if __name__ == '__main__':
    from funasr.utils.modelscope_param import modelscope_args
    params = modelscope_args(model="NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950", data_path = "./data")
    params.output_dir = "./checkpoint"
    params.data_dir = "./example_data/"
    params.batch_bins = 2000

    modelscope_finetune(params)
```



若想使用多卡进行微调训练，可按执行如下命令：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4  finetune.py > 1234.txt 2>&1
```

如果报P2P的错，可以加上NCCL_IGNORE_DISABLED_P2P=1，如下命令：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_IGNORE_DISABLED_P2P=1  python -m torch.distributed.launch --nproc_per_node 4  finetune.py > 1234.txt 2>&1
```




## 数据评估及结果（论文）
beam=20，CER tool：https://github.com/yufan-aslp/AliMeeting 

|        model        | Para (M) | Data (hrs) | Eval (CER%) | Test (CER%) |
|:-------------------:|:---------:|:---------:|:---------:| :---------:|
| MFCCA | 45   |   917  |   16.1   | 17.5   |

## 数据评估及结果（modelscope）

beam=10

with separating character (src)

|        model        | Para (M) | Data (hrs) | Eval_sp (CER%) | Test_sp (CER%) | 
|:-------------------:|:---------:|:---------:|:---------:| :---------:|
| MFCCA | 45   |   917  |   17.1   | 18.6   |

without separating character (src)

|        model        | Para (M) | Data (hrs) | Eval_nosp (CER%) | Test_nosp (CER%) | 
|:-------------------:|:---------:|:---------:|:---------:| :---------:|
| MFCCA | 45   |   917  |   16.4   | 18.0   |


## 可能的偏差

考虑到CER计算代码存在差异，解码beam论文是20，这边为了加快解码速度设置为10，会对CER的数据带来一定的差异（<0.5%）。


## 相关论文以及引用信息

```BibTeX
@inproceedings{yu2022mfcca,
  title={MFCCA:Multi-Frame Cross-Channel attention for multi-speaker ASR in Multi-party meeting scenario},
  author={Fan Yu, Shiliang Zhang, Pengcheng Guo, Yuhao Liang, Zhihao Du, Yuxiao Lin, Lei Xie},
  booktitle={Proc. SLT},
  pages={144--151},
  year={2023},
  organization={IEEE}
}
```