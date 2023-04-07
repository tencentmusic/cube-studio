import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class SPEECH_SAMBERT_HIFIGAN_TTS_LUNA_EN_GB_16K_Model(Model):
    # 模型基础信息定义
    name='speech-sambert-hifigan-tts-luna-en-gb-16k'   # 该名称与目录名必须一样，小写
    label='语音合成-英式英文-通用领域-16k-发音人Luna'
    describe="本模型是一种应用于参数TTS系统的后端声学模型及声码器模型。其中后端声学模型的SAM-BERT,将时长模型和声学模型联合进行建模。声码器在HIFI-GAN开源工作的基础上，我们针对16k, 48k采样率下的模型结构进行了调优设计，并提供了基于因果卷积的低时延流式生成和chunk流式生成机制，可与声学模型配合支持CPU、GPU等硬件条件下的实时流式合成。"
    field="听觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "3558"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/speech_sambert-hifigan_tts_luna_en-gb_16k/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='input', label='文本',describe='文本',default='',validators=Validator(max=1000))
    ]

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "input": "How is the weather in beijing?"
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
        
        self.p = pipeline('text-to-speech', 'damo/speech_sambert-hifigan_tts_luna_en-gb_16k')

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop()
    def inference(self,input,**kwargs):
        result = self.p(input)

        from modelscope.outputs import OutputKeys
        import random

        save_path = f'result/result{random.randint(1, 1000)}.wav'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)

        wav = result[OutputKeys.OUTPUT_WAV]
        with open(save_path, 'wb') as f:
            f.write(wav)

        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back = [
            {
                "audio": save_path
            }
        ]
        return back

model=SPEECH_SAMBERT_HIFIGAN_TTS_LUNA_EN_GB_16K_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(input='How is the weather in beijing?')  # 测试
# print(result)

# 模型启动web时使用
if __name__=='__main__':
    model.run()

# 模型大小 900M
# 模型运行速度  v100 gpu  占用1.5G 显存, 示例输入，耗时0.6s