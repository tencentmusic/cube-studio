import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type,Validator

import pysnooper
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch




class Whisper_large_Model(Model):
    # 模型基础信息定义
    name='whisper-large'
    label='asr+翻译成英文'
    describe="用于语音识别和翻译任务，能够将语音音频转录为所用语言 (ASR) 的文本，并翻译成英语"
    field="听觉"
    scenes="语种识别"
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址

    inference_inputs = [
        Field(type=Field_type.audio, name='audio_path', label='待识别的语音文件',describe='待识别文件')
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "audio_path": "test.wav"
            }
        }
    ]

    # 加载模型
    # @pysnooper.snoop(depth=2)
    def load_model(self,model_dir=None,**kwargs):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2",use_auth_token=os.getenv('HUGGINGFACE_TOKEN',None))
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2",use_auth_token=os.getenv('HUGGINGFACE_TOKEN',None))



    # 推理
    @pysnooper.snoop()
    def inference(self,audio_path):
        new_result=[]

        back=[
            {
                "html":"<br>".join(new_result)
            }
        ]
        return back

model=Whisper_large_Model()
# model.load_model()
# result = model.inference(audio_path='test.wav')  # 测试
# print(result)

if __name__=='__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py inference --arg1 xx --arg2 xx
    # python app.py web
    model.run()


