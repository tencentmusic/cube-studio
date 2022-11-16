import base64
import io, sys, os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server, Field, Field_type

import pysnooper
import os
import argparse, os, re
import datetime
import time

# speech所需内容，可根据需要注释无需的加载项
# 语音识别
from paddlespeech.cli.asr.infer import ASRExecutor
# 语音合成
from paddlespeech.cli.tts.infer import TTSExecutor
# 声音场景分类
from paddlespeech.cli.cls.infer import CLSExecutor
# 语音翻译
from paddlespeech.cli.st.infer import STExecutor


class Speech_Model(Model):
    # 模型基础信息定义
    name = 'paddle-speech'
    label = '语音处理'
    describe = "涵盖功能有语音转文字，文字转语音，语音翻译，语音场景识别"
    field = "智能识别"
    scenes = "语音处理"
    status = 'online'
    version = 'v20221114'
    doc = 'https://github.com/PaddlePaddle/PaddleSpeech'  # 'https://帮助文档的链接地址'
    # pic = 'https://images.nightcafe.studio//assets/stable-tile.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址

    inference_resource = {
        "resource_gpu": "1"
    }

    inference_inputs = [
        Field(type=Field_type.text, name='text', label='语音合成文本',
              describe='输入文本', default='cube studio 是个云原生一站式机器学习平台，欢迎大家体验！'),
        Field(type=Field_type.audio, name='voice', label='语音文件',
              describe='可支持上传一个语音文件，在下方选择一种想要进行的操作吧~'),
    ]
    web_examples = [
        {
            "label": "语音合成",
            "data": {
                "work": ['语音分类', '语音识别', '...'],
                "text": '今天天气不错',
                "name": 'a.wav',
            }
        }
    ]

    def __init__(self):
        self.st = None
        self.asr = None
        self.tts = None
        self.cls = None

    # 加载模型
    # @pysnooper.snoop()
    def load_model(self):
        self.cls = CLSExecutor()  # 语音分类
        self.tts = TTSExecutor()  # 语音合成
        self.asr = ASRExecutor()  # 语音识别
        self.st = STExecutor()  # 语音翻译

    # 推理
    @pysnooper.snoop()
    def inference(self, **kwargs):
        """
        kwargs例子：
            {
            "label": "语音合成",
            "data": {
                "work": ['语音分类', '语音识别', '...'],
                "text": '今天天气不错',
                "file_name": 'a.wav',
                }
            }
        """
        result = ''
        file_PATH = ''
        if kwargs['label'] == '语音合成':
            tts = self.tts
            file_name = f"{kwargs['data']['name']}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-output.wav"
            tts(text=kwargs['data']['text'], output=file_name)
            file_PATH = file_name
        elif kwargs['label'] == '语音内容识别':
            # file = open(kwargs['data']['name'], "wb")  # 写入二进制文件
            # text = base64.b64decode(kwargs['data']['text'])  # 进行解码
            # file.write(text)
            if '语音分类' in kwargs['data']['works']:
                cls = self.cls
                result += '语音分类结果： '
                result += cls(audio_file=kwargs['data']['file_name']) + '\r\n'
            if '语音识别' in kwargs['data']['works']:
                asr = self.asr
                result += '语音识别结果： '
                result += asr(audio_file=kwargs['data']['file_name']) + '\r\n'
            if '语音翻译' in kwargs['data']['works']:
                st = self.st
                result += '语音翻译结果： '
                result += ''.join(st(audio_file=kwargs['data']['file_name'])) + '\r\n'
        return file_PATH, result


model = Speech_Model()
# model.load_model()
# result = model.inference(prompt='a photograph of an astronaut riding a horse',device='cpu')  # 测试
# print(result)

# 启动服务
server = Server(model=model)
server.server(port=8080)
