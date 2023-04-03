import copy
import os,sys,json,time,random,io,base64
import argparse
import base64
import os
import datetime
import logging

import numpy

from ..utils import py_shell
import enum
import pysnooper
import os, sys
Field_type = enum.Enum('Field_type', ('int','double','json','text', 'image', 'audio', 'video', 'stream', 'text_select', 'image_select','audio_select','video_select','capture'))

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
    images=''

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

        # 校验字段
        if self.field not in ['机器视觉','听觉','自然语言','强化学习','图论','通用']:
            raise "field not valid，one of ['机器视觉','听觉','自然语言','强化学习','图论','通用']"

        self.doc = 'https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/%s'%self.name
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
            "inference": self.inference_config,
            'images':self.images
          }
        info["train"]={}
        if self.inference_resource.get('resource_memory',"0")!='0':
            info["inference"]['resource_memory']=self.inference_resource.get('resource_memory',"0")
        if self.inference_resource.get('resource_cpu',"0")!='0':
            info["inference"]['resource_cpu']=self.inference_resource.get('resource_cpu',"0")
        if self.inference_resource.get('resource_gpu',"0")!='0':
            info["inference"]['resource_gpu']=self.inference_resource.get('resource_gpu',"0")
            info["train"]['resource_gpu'] = self.inference_resource.get('resource_gpu', "0")
        if self.train_inputs:
            info["train"]={
                "job_template_args": {
                    "参数": {}
                }
            }
            for input in self.train_inputs:
                info["train"]['job_template_args']['参数']['--'+input.name.lstrip('-')]={
                    "type":"str",
                    "item_type":"str",
                    "label":input.label,
                    "require":1,
                    "choice":input.choices,
                    "range":"",
                    "default":input.default,
                    "placeholder":"",
                    "describe":input.describe,
                    "editable":1,
                    "condition":""
                }


        file=open('info.json',mode='w')
        file.write(json.dumps(info,indent=4,ensure_ascii=False))
        file.close()

    def init_args(self):
        main_parser = argparse.ArgumentParser(prog='PROG')
        task_type_subparsers = main_parser.add_subparsers(title="启动类型", dest="task_type")

        train_parser = task_type_subparsers.add_parser("train", help="启动训练")
        web_parser = task_type_subparsers.add_parser("web", help="启动web界面")
        inference_parser = task_type_subparsers.add_parser("inference", help="启动推理")
        download_model_parser = task_type_subparsers.add_parser("download_model", help="下载模型")
        # 训练子命令的参数
        for train_arg in self.train_inputs:
            train_parser.add_argument('--'+train_arg.name, type=str, help=train_arg.label,default=train_arg.default)
        # 推理子命令的参数
        for inference_arg in self.inference_inputs:
            inference_parser.add_argument('--'+inference_arg.name, type=str, help=inference_arg.label,default=inference_arg.default)
        # web子命令的参数
        web_parser.add_argument('--save_model_dir', type=str, help='load_mode函数的参数',default='')

        input = vars(main_parser.parse_args())
        task_type = input.get('task_type','web')
        kwargs = copy.deepcopy(input)
        if 'task_type' in kwargs:
            del kwargs['task_type']

        return task_type,input,kwargs

    def config(self,key,value=None):
        AIHUB_MODEL_CONFIG_PATH = os.getenv('AIHUB_MODEL_CONFIG_PATH', '')
        config = {}
        if AIHUB_MODEL_CONFIG_PATH:
            try:
                config = json.load(open(AIHUB_MODEL_CONFIG_PATH, mode='w'))
            except:
                config = {}

            if value:
                config[key] = value
                json.dump(config, open(AIHUB_MODEL_CONFIG_PATH, mode='w'), indent=4, ensure_ascii=False)
            else:
                return config.get(key,None)

    # @pysnooper.snoop()
    def run(self):
        task_type, input,kwargs = self.init_args()

        print(task_type,input,kwargs)
        if task_type == 'train':
            print('启动训练')
            save_model_dir = self.train(**kwargs)
            self.config('save_model_dir',save_model_dir)
        elif task_type == 'download_model':
            print('下载模型')
            self.download_model()

        elif task_type == 'inference':
            print('启动推理')
            # 先从启动参数中读取，再从配置文件中读取
            save_model_dir = input.get('save_model_dir',self.config('save_model_dir'))

            self.load_model(save_model_dir)
            if save_model_dir in kwargs:
                del kwargs['save_model_dir']
            result = self.inference(**kwargs)  # 测试
            print(result)
        else:
            print('启动web服务')
            from .web.server import Server
            if os.getenv('REQ_TYPE', 'synchronous') == 'synchronous':
                # 先从启动参数中读取，再从配置文件中读取
                self.load_model(save_model_dir=kwargs.get('save_model_dir',self.config('save_model_dir')))
            server = Server(model=self)
            server.server(port=8080)

    # 配置数据集，在分布式训练时自动进行分配
    def set_dataset(self,**kwargs):
        pass

    # 训练的入口函数，将用户输入参数传递
    def train(self,save_model_dir:str, **kwargs):
        print('train函数接收到参数：',kwargs,'但是此模型并未实现train逻辑')


    # 推理前加载模型
    def load_model(self,save_model_dir=None,**kwargs):
        print('load_model函数接收到参数：', kwargs, '但是此模型并未实现load_model逻辑')

    def download_model(self):
        print('开始下载模型')

    # 同步推理函数
    def inference(self,**kwargs):
        print('inference函数接收到参数：', kwargs, '但是此模型并未实现inference逻辑')
    # 接受rtsp流的推理
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        print('rtsp inference函数接收到参数：', kwargs, '但是此模型并未实现rtsp inference逻辑')
        return img
    # 批推理
    def batch_inference(self,**kwargs):
        print('batch_inference函数接收到参数：', kwargs, '但是此模型并未实现batch_inference逻辑')

