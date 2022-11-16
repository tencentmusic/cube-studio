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

from paddlespeech.cli.tts.infer import TTSExecutor


class Speech_Tts_Model(Model):
    # 模型基础信息定义
    name = 'paddle-speech-tts'
    label = '文字转语音'
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
        Field(type=Field_type.text, name='text', label='语音转文本',
              describe='输入文本', default='cube studio 是个云原生一站式机器学习平台，欢迎大家体验！'),
        # Field(type=Field_type.text, name='spk_id', label='说话人ID',
        #       describe='0-283可选，不一样的ID会带来不一样的声音', default=0),
        Field(type=Field_type.text_select, name='spk_id', label='说话人', default='1',
              choices=['1', '2', '3', '4', '5'])

    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "text": "今天天气不错",
                "spk_id": '0',
            }
        }
    ]

    def __init__(self):
        self.tts = None

    # 加载模型
    # @pysnooper.snoop()
    def load_model(self):
        self.tts = TTSExecutor()  # 语音合成

    # 推理
    @pysnooper.snoop()
    def inference(self, text, spk_id=0):
        tts = self.tts
        os.makedirs('result', exist_ok=True)
        if spk_id:
            spk_id = int(spk_id) - 1 if int(spk_id) > 0 else int(spk_id)
        else:
            spk_id = 0
        file_name = f"result/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-output.wav"
        tts(text=text, output=file_name, am='fastspeech2_mix', voc='hifigan_csmsc', lang='mix', spk_id=spk_id)
        back = [
            {
                'text': text,
                'audio': file_name
            }
        ]
        return back


model = Speech_Tts_Model()
# model.load_model()
# result = model.inference('hello,你好，hi。yesterday')  # 测试
# print(result)

# 启动服务
server = Server(model=model)
server.server(port=8080)
