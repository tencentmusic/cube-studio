import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class SPEECH_MFCCA_ASR_ZH_CN_16K_ALIMEETING_VOCAB4950_Model(Model):
    # 模型基础信息定义
    name='speech-mfcca-asr-zh-cn-16k-alimeeting-vocab4950'   # 该名称与目录名必须一样，小写
    label='MFCCA多通道多说话人语音识别-中文-AliMeeting-16k-离线'
    describe="考虑到麦克风阵列不同麦克风接收信号的差异，该模型采用了一种多帧跨通道注意力机制，该方法对相邻帧之间的跨通道信息进行建模，以利用帧级和通道级信息的互补性。此外，还引入了一种多层卷积模块以融合多通道输出和一种通道掩码策略以解决训练和推理之间的音频通道数量不匹配的问题。在ICASSP2022 M2MeT竞赛上发布的真实会议场景语料库AliMeeting上进行了相关实验，该多通道模型在Eval和Test集上比单通道模型CER分别相对降低了39.9%和37.0%。此外，在同等的模型参数和训练数据下，本文提出的模型获得的识别性能超越竞赛期间最佳结果，在AliMeeting上实现了目前最新的SOTA性能。"
    field="听觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "425"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.audio, name='input', label='音频',describe='音频',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "input": "/mnt/workspace/.cache/modelscope/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/example/asr_example_mc.wav"
            }
        }
    ]

    # 训练的入口函数，此函数会自动对接pipeline，将用户在web界面填写的参数传递给该方法
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        pass
        # 训练的逻辑
        # 将模型保存到save_model_dir 指定的目录下


    # 加载模型，所有一次性的初始化工作可以放到该方法下。注意save_model_dir必须和训练函数导出的模型结构对应
    def load_model(self,save_model_dir=None,**kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        self.p = pipeline('auto-speech-recognition', 'NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,input,**kwargs):
        result = self.p(input)

        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back=[
            {
                "image": 'result/aa.jpg',
                "text": '结果文本',
                "video": 'result/aa.mp4',
                "audio": 'result/aa.mp3',
                "markdown":''
            }
        ]
        return back

model=SPEECH_MFCCA_ASR_ZH_CN_16K_ALIMEETING_VOCAB4950_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(input='/mnt/workspace/.cache/modelscope/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/example/asr_example_mc.wav')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()