
import flask,os,sys,json,time,random,io,base64

import base64
import os
import numpy as np
import datetime
import logging
import cv2
from flask import jsonify,request

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

    init_shell=''

    def __init__(self,init_shell=True):
        print('begin init shell')
        if init_shell and self.init_shell:
            os.system('bash %s'%self.init_shell)

        print('begin load model')
        self.load_model()


    # 加载模型
    def load_model(self,**kwargs):
        pass

    # 配置数据集
    def set_dataset(self,**kwargs):
        pass

    # 训练函数
    def train(self,**kwargs):
        pass

    # 同步推理函数
    def inference(self,**kargs):
        pass

    # 批推理
    def batch_inference(self,**kwargs):
        pass

    # 测试
    def test(self,**kwargs):
        pass
