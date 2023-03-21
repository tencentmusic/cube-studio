
# ECAPA-TDNN说话人模型
ECAPA-TDNN模型是基于时延神经网络构建的说话人模型，由于识别性能优异，已经被广泛使用在说话人识别领域中。该模型还可以用于说话人确认、说话人日志等任务。
## 模型结构简述
ECAPA-TDNN在传统的TDNN模型上有3种改进。第一，融合了一维的Res2Net层和Squeeze-and-Excitation模块，对特征channel之间的关系进行建模。第二，融合多个层级特征，同时利用网络浅层和深层的信息。第三，采用了基于attention机制的pooling层，生成基于全局attention的说话人特征。
<div align=center>
<img src="https://modelscope.cn/api/v1/models/damo/speech_ecapa-tdnn_sv_en_voxceleb_16k/repo?Revision=master&FilePath=ecapa_tdnn.jpg&View=true" width="300" />
</div>
更详细的信息见：[论文](https://arxiv.org/abs/2005.07143)

## 训练数据
本模型使用公开的英文说话人数据集VoxCeleb2进行训练，可以对16k采样率的英文音频进行说话人识别。
## 模型效果评估
- 选择EER、minDCF作为客观评价指标。
- 在VoxCeleb1-O测试集上，EER = 0.862，minDCF(p_target=0.01, c_miss=c_fa=1) = 0.094。

# 如何快速体验模型效果【开发中】
## 在线体验【开发中】
在页面右侧，可以在“在线体验”栏内看到我们预先准备好的示例音频，点击播放按钮可以试听，点击“执行测试”按钮，会在下方“测试结果”栏中显示相似度得分(范围为[-1,1])和是否判断为同一个人。如果您想要测试自己的音频，可点“更换音频”按钮，选择上传或录制一段音频，完成后点击执行测试，识别内容将会在测试结果栏中显示。
## 在Notebook中体验【开发中】
对于有开发需求的使用者，特别推荐您使用Notebook进行离线处理。先登录ModelScope账号，点击模型页面右上角的“在Notebook中打开”按钮出现对话框，首次使用会提示您关联阿里云账号，按提示操作即可。关联账号后可进入选择启动实例界面，选择计算资源，建立实例，待实例创建完成后进入开发环境，输入api调用实例。
```python
from modelscope.pipelines import pipeline
sv_pipline = pipeline(
    task='speaker-verification',
    model='damo/speech_ecapa-tdnn_sv_en_voxceleb_16k'
)
speaker1_a_wav = 'https://modelscope.cn/api/v1/models/damo/speech_ecapa-tdnn_sv_en_voxceleb_16k/repo?Revision=master&FilePath=examples/speaker1_a_en_16k.wav'
speaker1_b_wav = 'https://modelscope.cn/api/v1/models/damo/speech_ecapa-tdnn_sv_en_voxceleb_16k/repo?Revision=master&FilePath=examples/speaker1_b_en_16k.wav'
speaker2_a_wav = 'https://modelscope.cn/api/v1/models/damo/speech_ecapa-tdnn_sv_en_voxceleb_16k/repo?Revision=master&FilePath=examples/speaker2_a_en_16k.wav'
# 相同说话人语音
result = sv_pipline([speaker1_a_wav, speaker1_b_wav])
print(result)
# 不同说话人语音
result = sv_pipline([speaker1_a_wav, speaker2_a_wav])
print(result)
# 可以自定义得分阈值来进行识别
result = sv_pipline([speaker1_a_wav, speaker2_a_wav], thr=0.6)
print(result)
```
## 训练和测试自己的ECAPA-TDNN模型【开发中】
正在开源中...

# 相关论文以及引用信息

```BibTeX
@article{ecapa_tdnn,
  title={ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification},
  author={Brecht, Desplanques and Jenthe, Thienpondt and Kris, Demuynck},
  journal={arXiv preprint arXiv:2005.07143},
}
```
