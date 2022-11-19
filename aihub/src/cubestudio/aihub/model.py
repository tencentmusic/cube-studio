
import os,sys,json,time,random,io,base64

import base64
import os
import datetime
import logging
from ..util import py_shell
import enum
import os, sys
Field_type = enum.Enum('Field_type', ('int','double','json','str','text', 'image', 'audio', 'video', 'stream', 'text_select', 'image_select','audio_select','video_select'))

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
        self.validators = validators if validators else Validator()


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

    # 训练数据集
    dataset_config = {}
    # notebook相关信息
    notebook_config={
        "jupyter": [],
        "appendix": []
    }
    train_config={}
    automl_config={}
    inference_config={}


    # 开发notebook
    notebook_jupyter=[]

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
    web_examples=[]
    inference_resource={
        "resource_memory":"0",
        "resource_cpu":"0",
        "resource_gpu":"0"
    }
    inference_env={
        "APP_NAME":name
    }

    def __init__(self):
        # 生成info.json文件
        info={
            "doc": self.doc,
            "field": self.field,
            "scenes": self.scenes,
            "type": "dateset,notebook,train,inference",
            "name": self.name,
            "status": self.status,
            "version": self.version,
            "uuid": self.name+"-"+self.version,
            "label": self.label,
            "describe": self.describe,
            "pic": self.pic,
            "hot": "1",
            "price": "0",
            "dataset":self.dataset_config,
            "notebook":self.notebook_config,
            "train": self.train_config,
            "inference": self.inference_config
          }

        if self.inference_resource.get('resource_memory',"0")!='0':
            info["inference"]['resource_memory']=self.inference_resource.get('resource_memory',"0")
        if self.inference_resource.get('resource_cpu',"0")!='0':
            info["inference"]['resource_cpu']=self.inference_resource.get('resource_cpu',"0")
        if self.inference_resource.get('resource_gpu',"0")!='0':
            info["inference"]['resource_gpu']=self.inference_resource.get('resource_gpu',"0")

        file=open('info.json',mode='w')
        file.write(json.dumps(info,indent=4,ensure_ascii=False))
        file.close()

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

