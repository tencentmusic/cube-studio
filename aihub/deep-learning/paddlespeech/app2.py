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
    name = 'paddle-speech'
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
        Field(type=Field_type.text, name='spk_id', label='说话人ID',
              describe='0-283可选，不一样的ID会带来不一样的声音', default=0),
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "text": "今天天气不错",
                "spk_id": 0,
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
        file_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-output.wav"
        if spk_id:
            if str(spk_id).isdigit():
                if 0<=int(spk_id)<=283:
                    spk_id=int(spk_id)
                else:
                    back = [{'text':'警告：数字需要在0~283之间哦~'}]
                    return back
            else:
                back = [{'text':'警告：说话人仅支持数字哦~'}]
                return back
        else:
            spk_id = 0
        tts(text=text, output=file_name, am='fastspeech2_mix', voc='hifigan_csmsc',lang='mix', spk_id=spk_id)
        back = [
            {
                'text': text,
                'audio': file_name
            }
        ]
        return back


model = Speech_Tts_Model()
#model.load_model()
#result = model.inference('hello,你好，hi。yesterday')  # 测试
#print(result)

# 启动服务
server = Server(model=model)
server.server(port=8080)
