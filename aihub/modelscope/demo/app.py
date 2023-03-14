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
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,{{app.inference_fun_args}},**kwargs):
        {{app.inference_fun}}
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

model={{app.name.upper().replace("-","_")}}_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference({{app.inference_fun_args_value}})  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()

