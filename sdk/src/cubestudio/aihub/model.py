
import os,sys,json,time,random,io,base64

import base64
import os
import datetime
import logging
from ..util import py_shell
import enum
import os, sys
Field_type = enum.Enum('Field_type', ('json','str','text', 'image', 'video', 'stream', 'text_select', 'image_select', 'text_multi', 'image_multi'))

class Field():
    def __init__(self,type:Field_type,name:str,label:str,describe='',choices=[],default='',validators=[]):
        self.type=type
        self.name=name
        self.label=label
        self.describe=describe
        self.choices=choices
        self.default=default
        self.validators=validators

    def to_json(self):
        return {
            "type":str(self.type.name),
            "name":self.name,
            "label":self.label,
            "describe":self.describe,
            "choices":self.choices,
            "default":self.default,
            "validators":self.validators
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
    base_images='ccr.ccs.tencentyun.com/cube-studio/aihub:base'

    # 训练的输入
    train_inputs=[]
    inference_inputs=[]

    def __init__(self,init_shell=True):
        print('begin init shell')
        if init_shell and self.init_shell:
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
