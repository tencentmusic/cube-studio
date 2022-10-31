
import os,sys,json,time,random,io,base64

import base64
import os
import datetime
import logging
from ..util import py_shell
import enum
import os, sys
Field_type = enum.Enum('Field_type', ('int','double','json','str','text', 'image', 'audio', 'video','video_multi', 'stream', 'text_select', 'image_select'))

class Validator():
    def __init__(self,regex='',min=1,max=1,required=True):
        self.regex = regex
        self.min=min
        self.max=max
        self.regex = regex
        self.required = required

    def to_json(self):
        return {
            "regex": self.regex,
            "min": self.min,
            "max": self.max,
            "required":self.required
        }

        pass
class Field():
    def __init__(self,type:Field_type,name:str,label:str,validators:Validator=None,describe='',choices=[],default=''):
        self.type=type
        self.name=name
        self.label=label
        self.describe=describe
        self.choices=choices
        self.default=default
        self.validators = validators


    def to_json(self):
        vals=[]
        if self.validators:
            if self.validators.regex:
                vals.append(
                    {
                        "type":"Regexp",   # # Regexp Length  DataRequired,
                        "regex":self.validators.regex
                    }
                )
            if self.validators.max or self.validators.min:
                vals.append(
                    {
                        "type": "Length",  # # Regexp Length  DataRequired,
                        "min": self.validators.min,
                        "max":self.validators.max
                    }
                )
            if self.validators.required:
                vals.append(
                    {
                        "type": "DataRequired"  # # Regexp Length  DataRequired,
                    }
                )

        return {
            "type":str(self.type.name),
            "name":self.name,
            "label":self.label,
            "describe":self.describe,
            "values":[{"id":choice[choice.rindex('/')+1:] if 'http' in choice else choice,"value":choice} for choice in self.choices],
            # "values": [{"value": choice[choice.rindex('/') + 1:] if 'http' in choice else choice, "id": choice} for choice in self.choices],
            # "choices": [[choice,choice] for choice in self.choices],
            "maxCount":self.validators.max if self.validators else None,
            "default":self.default,
            "validators":vals
        }


class Model():

    # 模型的基础信息
    name = 'demo'
    doc ='https://github.com/tencentmusic/cube-studio'
    field = "机器视觉"
    scenes="图像分类"
    status="online"
    version="v20221001"
    uuid="2022100101"
    label="模型的简短描述"
    describe="这里可以添加模型的详细描述，建议在10~100字内"
    pic="http://xx.xx.jpg"
    price="0"
    notebook_config={}
    job_template_config={}
    inference_config={}
    automl_config={}
    # 基础运行环境
    init_shell=''
    base_images='ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9'

    # notebook相关信息
    notebook={
        "jupyter": [],
        "appendix": []
    }

    # 训练相关
    train_inputs=[
        # Field
    ]
    train_resource={
        "resource_memory":"0",
        "resource_cpu":"0",
        "resource_gpu":"0"
    }
    train_env={
        "APP_NAME":name
    }

    # 推理相关
    inference_inputs=[
        # Field
    ]
    inference_resource={
        "resource_memory":"0",
        "resource_cpu":"0",
        "resource_gpu":"0"
    }
    inference_env={
        "APP_NAME":name
    }

    def __init__(self,init_shell=False):
        if init_shell and self.init_shell:
            print('begin init shell')
            py_shell.exec('bash %s'%self.init_shell)

    # 配置数据集，在分布式训练时自动进行分配
    def set_dataset(self,**kwargs):
        pass

    # 训练函数
    def train(self,**kwargs):
        pass

    # 推理前加载模型
    def load_model(self,**kwargs):
        pass

    # 同步推理函数
    def inference(self,**kargs):
        pass

    # 批推理
    def batch_inference(self,**kwargs):
        pass
