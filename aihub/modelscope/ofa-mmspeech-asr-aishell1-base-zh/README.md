
``
注：请使用modelscope==1.2.0以上版本
``
# MMSpeech

## ASR是什么
ASR(Automatic Speech Recognition)语音识别技术，是一种将人的语音转换为文本的技术。概念虽然简单，但是实际算法比较复杂，真正的实用化就会更加复杂。ASR的评估指标一般是WER(Words Error Rate)，显然没有达到一定阈值的模型都没有应用场景。

## MMSpeech是什么
**MMSpeech**是达摩院自研，于**2022年12月份新鲜出炉**的语音预训练模型，基于[OFA架构](https://modelscope.cn/docs/OFA%20Tutorial) ，能够充分利用无标注文本，显著的降低了字错误率，对比HuBert和Wav2Vec等知名模型的中文版本，在标准benchmark AIShell1的验证集/测试集上的字错误率降低了**48.3%/42.4%**，效果达到 **1.6%/1.9%**，远超SOTA 3.1%/3.3%。

目前已经上线4个OFA-MMSpeech模型，其他3个是：
- [预训练Base模型](https://modelscope.cn/models/damo/ofa_mmspeech_pretrain_base_zh/summary)
- [预训练Large模型](https://modelscope.cn/models/damo/ofa_mmspeech_pretrain_large_zh/summary)
- [AIShell1微调Large模型](https://modelscope.cn/models/damo/ofa_mmspeech_asr_aishell1_large_zh/summary)

## 方法描述

MMSpeech是一种针对中文语音识别任务的预训练方法，该方法利用了大量无标注的语音和文本数据，并设计了五个语音/文本任务在统一的encoder-decoder模型框架下进行多任务学习。

相比以往的预训练方法，MMSpeech具有两点优势：
一个是我们使用了大量的无标注文本数据（一共292G）来提升语音识别预训练效果。这不同于语音单模态预训练方法（Wav2Vec、HuBERT、Data2Vec、WavLM等）不使用文本数据，或其他语音-文本联合预训练方法（SpeechT5、STPT）使用较少的文本数据（1.8G），我们充分探索了文本数据对语音识别预训练的价值，并证明能够带来很大的提升。
另一方面，MMSpeech是专门针对中文语音场景优化的预训练方法。以往的预训练方法大多是在英文数据上进行探究，而相比英文，中文是表意语音，语音和文本模态之间的差异更大，这导致语音/文本任务在共享统一模型时面临困难。我们提出将音素（phone，这里用的是拼音）这个紧密联系语音和文本的模态引入到预训练过程中，来缓解语音和文本模态之间的差异问题，并且实验证明音素的引入可以使文本数据发挥更大的价值，并提升预训练效果。

详细内容见论文介绍：[https://arxiv.org/abs/2212.00500](https://arxiv.org/abs/2212.00500)

<div align=center>
<img src="fig/mmspeech.png" width="700" height="400"/>
</div>

具体地，MMSpeech的训练流程如上图所示，总共有五个任务，分别是phone-to-text（P2T）、speech-to-code（S2C）、masked speech prediction（MSP）、phoneme prediction（PP） 和 speech-to-text（S2T）。

其中，phone-to-text、speech-to-code这两个任务是分别利用了无标注的文本数据和无标注的语音数据，构造了伪pair数据帮助encoder-decoder进行自监督学习。同时，考虑到过去很多针对encoder的语音预训练的方法（比如Wav2Vec）通过无标注语音数据训练encoder得到一个好的语音表示能够有效提升语音任务效果，MMSpeech也引入了masked speech prediction、phoneme prediction这两个任务利用无标注语音数据对encoder进行预训练。最后，我们还将下游语音识别任务speech-to-text也引入到多任务学习中，能够进一步提升模型效果。

## 相关模型
本模型为下表中的MMSpeech-Base的Finetune模型。

### 预训练模型
| Model          | Model Size | Unlabeled Speech | Unlabeled Text |  labeled  |                                                Pre-Training                                                |                                                    Fine-Tuning                                                     |
|:---------------|:----------:|:----------------:|:--------------:|:---------:|:----------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|
| MMSpeech-Base  |    210M    |    AISHELL-2     |   M6-Corpus    | AISHELL-1 | [ofa_mmspeech_pretrain_base_zh](https://modelscope.cn/models/damo/ofa_mmspeech_pretrain_base_zh/summary)   |  [ofa_mmspeech_asr_aishell1_base_zh](https://modelscope.cn/models/damo/ofa_mmspeech_asr_aishell1_base_zh/summary)  |
| MMSpeech-Large |    609M    |   WenetSpeech    |   M6-Corpus    | AISHELL-1 | [ofa_mmspeech_pretrain_large_zh](https://modelscope.cn/models/damo/ofa_mmspeech_pretrain_large_zh/summary) | [ofa_mmspeech_asr_aishell1_large_zh](https://modelscope.cn/models/damo/ofa_mmspeech_asr_aishell1_large_zh/summary) |

## 模型推理
输入音频文件，输出目标文字。模型支持任意采样率和通道数wav音频文件输入，推荐输入语音时长在20s以下。

#### api调用范例
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pretrained_model = "damo/ofa_mmspeech_asr_aishell1_base_zh"
pipe = pipeline(Tasks.auto_speech_recognition, model=pretrained_model, model_revision='v1.0.0')
result = pipe({"wav": "https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/speech/asr_example_ofa.wav"})
print(result) # 甚至出现交易几乎停滞的情况
```

## 数据评估及结果
在 AISHELL-1 的dev/test数据集上进行测试

| Model                   | dev(w/o LM) | dev(with LM) | test(w/o LM) | test(with LM) |
|:------------------------|:-----------:|:------------:|:------------:|:-------------:|
| MMSpeech-Base-Pretrain  |     2.5     |     2.3      |     2.6      |      2.3      |
| MMSpeech-Base-aishell1  |     2.4     |     2.1      |     2.6      |      2.3      |
| MMSpeech-Large-Pretrain |     2.0     |     1.8      |     2.1      |      2.0      |
| MMSpeech-Large-aishell1 |     1.8     |     1.6      |     2.0      |      1.9      |


## 相关论文以及引用
如果你觉得OFA-MMSpeech好用，喜欢我们的工作，欢迎引用：

```
@article{zhou2022mmspeech,
  author    = {Zhou, Xiaohuan and 
               Wang, Jiaming and 
               Cui, Zeyu and 
               Zhang, Shiliang and 
               Yan, Zhijie and 
               Zhou, Jingren and 
               Zhou, Chang},
  title     = {MMSpeech: Multi-modal Multi-task Encoder-Decoder Pre-training for Speech Recognition},
  journal   = {arXiv preprint arXiv:2212.00500},
  year      = {2022}
}
```

```
@article{wang2022ofa,
  author    = {Peng Wang and
               An Yang and
               Rui Men and
               Junyang Lin and
               Shuai Bai and
               Zhikang Li and
               Jianxin Ma and
               Chang Zhou and
               Jingren Zhou and
               Hongxia Yang},
  title     = {OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence
               Learning Framework},
  journal   = {CoRR},
  volume    = {abs/2202.03052},
  year      = {2022}
}
```