
# Paraformer模型介绍

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


## 如何快速体验模型效果

### 在线体验

在页面右侧，可以在“在线体验”栏内看到我们预先准备好的示例音频，点击播放按钮可以试听，点击“执行测试”按钮，会在下方“测试结果”栏中显示识别结果。如果您想要测试自己的音频，可点“更换音频”按钮，选择上传或录制一段音频，完成后点击执行测试，识别内容将会在测试结果栏中显示。

### 在Notebook中开发

对于有开发需求的使用者，特别推荐您使用Notebook进行离线处理。先登录ModelScope账号，点击模型页面右上角的“在Notebook中打开”按钮出现对话框。api调用方式可参考如下范例：

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_16k_pipline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer_asr_nat-aishell1-pytorch')

rec_result = inference_16k_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
```

如果输入音频为pcm格式，调用api时需要传入音频采样率参数audio_fs，例如：
```python
rec_result = inference_16k_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.pcm', audio_fs=16000)
```

## 如何训练自己的Paraformer模型？

本项目提供的Paraformer是基于AISHELL-1的识别模型，开发者可以基于此模型进一步利用MaaSlib的finetuning功能或者本项目对应的github代码仓库进一步进行模型的领域定制化。

### 基于github的模型训练和推理

FunASR框架支持魔搭社区开源的工业级的语音识别模型的training & finetuning，使得研究人员和开发者可以更加便捷的进行语音识别模型的研究和生产，目前已在github开源：https://github.com/alibaba-damo-academy/FunASR。

#### FunASR框架安装

- 安装FunASR和ModelScope

```sh
# Clone the repo:
git clone https://github.com/alibaba/FunASR.git

# Install Conda:
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
conda create -n funasr python=3.7
conda activate funasr

# Install Pytorch (version >= 1.7.0):
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch  # For more versions, please see https://pytorch.org/get-started/locally/

# Install ModelScope
pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

# Install other packages:
pip install --editable ./
```

接下来会以AISHELL-1数据集为例，介绍如何在FunASR框架中使用Paraformer进行训、微调和推理。
#### 训练
```sh
cd egs/aishell/paraformer
# 配置 run.sh 中参数
# CUDA_VISIBLE_DEVICES: 可用的gpu list
# gpu_num: 训练使用的gpu数量
# data_aishell: AISHELL-1原始数据路径
# feats_dir: 数据准备输出目录
# exp_dir: 模型输出路径
# tag: 模型保存后缀名
# 配置修改完成后，执行命令: 
bash run.sh --stage 0 --stop_stage 3
```
#### 微调：加载训练好的模型，采用私有或者开源数据进行模型训练。
```sh
cd egs/aishell/paraformer
# 配置 run.sh 中参数
# CUDA_VISIBLE_DEVICES: 可用的gpu list
# gpu_num: 训练使用的gpu数量
# feats_dir: 数据准备输出目录
# exp_dir: 模型输出路径
# tag: 模型保存后缀名
# init_param: 初始模型路径
# 配置修改完成后，执行命令: 
bash run.sh --stage 3 --stop_stage 3
```
#### 推理
```sh
cd egs/aishell/paraformer
# 配置 run.sh 中参数
# CUDA_VISIBLE_DEVICES: 可用的gpu list
# gpu_num: 训练使用的gpu数量
# njob: 一块gpu解码的线程数
# feats_dir: 数据准备输出目录
# exp_dir: 模型输出路径
# tag: 模型保存后缀名
# 配置修改完成后，执行命令: 
bash run.sh --stage 4 --stop_stage 4
```

### 基于MaaSlib的模型训练和推理

正在对接中，预计12月底完成接入，敬请期待。

## 数据评估及结果


|          model          |  dev(CER%)  |  test(CER%)  |   RTF    |
|:-----------------------:|:---------------:|:----------------:|:--------:|
|       Paraformer        |      4.66       |       5.11      |  0.0168  |


## 使用方式以及适用范围

运行范围
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

使用方式
- 直接推理：可以直接对输入音频进行解码，输出目标文字。
- 微调：加载训练好的模型，采用私有或者开源数据进行模型训练。

使用范围与目标场景
- 适合与离线语音识别场景，如录音文件转写，配合GPU推理效果更加，推荐输入语音时长在20s以下。


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
