import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class SPEECH_DFSMN_AEC_PSM_16K_Model(Model):
    # 模型基础信息定义
    name='speech-dfsmn-aec-psm-16k'   # 该名称与目录名必须一样，小写
    label='DFSMN回声消除-单麦单参考-16k'
    describe="支持音频通话场景的单通道回声消除模型算法。模型接受单通道麦克风信号和单通道参考信号作为输入，输出回声消除和残余抑制后的音频信号[1]。模型采用Deep FSMN结构，提取原始观测信号以及线性滤波后信号的Fbank特征作为输入，预测输出目标语音的Phase senstive mask。"
    field="听觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.png'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "6284"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/speech_dfsmn_aec_psm_16k/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.audio, name='nearend_mic', label='麦克风录制的音频',describe='麦克风录制的音频',default='',validators=None),
        Field(type=Field_type.audio, name='farend_speech', label='喇叭播放的音频',describe='喇叭播放的音频',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "nearend_mic": "/mnt/workspace/.cache/modelscope/damo/speech_dfsmn_aec_psm_16k/examples/nearend_mic1.wav",
                "farend_speech": "/mnt/workspace/.cache/modelscope/damo/speech_dfsmn_aec_psm_16k/examples/farend_speech1.wav"
            }
        },
        {
            "label": "示例2",
            "input": {
                "nearend_mic": "/mnt/workspace/.cache/modelscope/damo/speech_dfsmn_aec_psm_16k/examples/nearend_mic.wav",
                "farend_speech": "/mnt/workspace/.cache/modelscope/damo/speech_dfsmn_aec_psm_16k/examples/farend_speech.wav"
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
        
        self.p = pipeline('acoustic-echo-cancellation', 'damo/speech_dfsmn_aec_psm_16k')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,nearend_mic,farend_speech,**kwargs):
        result = self.p({"nearend_mic": nearend_mic, "farend_speech": farend_speech})

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

model=SPEECH_DFSMN_AEC_PSM_16K_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(nearend_mic='/mnt/workspace/.cache/modelscope/damo/speech_dfsmn_aec_psm_16k/examples/nearend_mic1.wav',farend_speech='/mnt/workspace/.cache/modelscope/damo/speech_dfsmn_aec_psm_16k/examples/farend_speech1.wav')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()