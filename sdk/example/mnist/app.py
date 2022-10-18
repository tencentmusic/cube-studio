import base64
import io,sys,os
root_dir = os.path.split(os.path.realpath(__file__))[0] + '/../../src/'
print(root_dir)
sys.path.append(root_dir)   # 将根目录添加到系统目录,才能正常引用common文件夹

import pysnooper
from PIL import ImageGrab, Image
import numpy
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.app import Server,Field,Field_type
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import run
import torch.distributed as dist
from torchvision import datasets, transforms

class Mnist_Model(Model):
    # 模型基础信息定义
    name='mnist'
    label='手写体识别'
    description="算法入门手写体识别示例"
    field="机器视觉"
    scenes="图像分类"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/mnist'
    pic='http://5b0988e595225.cdn.sohucs.com/images/20180404/9f62a1eea7054f8eaf4bc1c87168238b.png'
    # 运行基础环境脚本
    init_shell='init.sh'
    base_images = 'ccr.ccs.tencentyun.com/cube-studio/aihub:base'

    # 训练
    train_inputs = [
        Field(Field_type.str, name='modelpath', label='模型存储地址', describe='模型存储地址'),
        Field(Field_type.str, name='datapath', label='数据地址', describe='训练数据地址')
    ]
    # 推理输入
    inference_inputs = [
        Field(type=Field_type.image,name='img_file_path',label='待识别图片',describe='用于文本识别的原始图片')
    ]


    # 训练的入口函数，将用户输入参数传递
    def train(self,**kwargs):
        dist.init_process_group(backend='gloo')
        run(modelpath=kwargs['modelpath'], gpu=False, datapath=kwargs['datapath'])
        dist.destroy_process_group()

    # 推理前load模型
    def load_model(self,**kwargs):
        from mnist import Net
        self.model = Net()
        device = 'cpu'
        modelpath = '/mnt/admin/pytorch/model/model_cpu.dat'
        self.model.load_state_dict(torch.load(modelpath))
        self.model.to(device)
        self.model.eval()

    # 单数据推理函数
    # @pysnooper.snoop()
    def inference(self,img_file_path):

        pic=Image.open(img_file_path)
        transform_vaild = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        inputs = transform_vaild(pic).unsqueeze(0)
        inputs = inputs.to('cpu')
        outputs = self.model(inputs)
        outputs = F.softmax(outputs, dim=1)
        max = outputs[0][0]
        maxnum = 0
        for i, j in enumerate(outputs[0]):
            if j > max:
                max = j
                maxnum = i

        back=[{
            "text":maxnum
        }]
        return back

# 训练
model=Mnist_Model(init_shell=False)
model.train(modelpath='/model/',datapath='')

# 离线推理
model.load_model()
result = model.inference(img_file_path='mnist.png')  # 测试
print(result)
result = model.batch_inference(['mnist.png'])  # 测试
print(result)

# web服务
server = Server(model=model)
server.web_examples.append('mnist.png')
server.server(port=8080)
