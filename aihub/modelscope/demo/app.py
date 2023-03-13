import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server

import pysnooper
import os

class {{app.name.upper().replace("-","_")}}_Model(Model):
    # 模型基础信息定义
    name='{{app.name.lower()}}'   # 该名称与目录名必须一样，小写
    label='{{app.label}}'
    describe="{{app.describe}}"
    field="{{app.field}}"
    scenes="{{app.scenes}}"
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "{{app.hot}}"
    frameworks = "{{app.frameworks}}"
    doc = "{{app.url}}"

    train_inputs = {{app.train_inputs}}

    inference_inputs = {{app.inference_inputs}}

    inference_resource = {
        "resource_gpu": "{{app.resource_gpu}}"
    }

    web_examples={{app.web_examples}}

    # 训练的入口函数，将用户输入参数传递
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        {{app.train_fun}}


    # 加载模型
    def load_model(self,save_model_dir=None,**kwargs):
        {{app.load_model_fun}}

    # 推理
    # @pysnooper.snoop()
    def inference(self,{{app.inference_fun_args}},**kwargs):
        {{app.inference_fun}}
        back=[
            {
                "image": result_img,
                "text": result_text,
                "video": result_video,
                "audio": result_audio,
                "markdown":result_markdown
            }
        ]
        return back

model={{app.name.upper().replace("-","_")}}_Model()

# model.load_model()
# result = model.inference(arg1='测试输入文本',arg2='test.jpg')  # 测试
# print(result)

if __name__=='__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py inference --arg1 xx --arg2 xx
    # python app.py web --save_model_dir xx
    # python app.py download_model 用于再构建镜像下载一些预训练模型
    model.run()

