import base64
import io,sys,os
import shutil

from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type


import pysnooper
from deoldify.visualize import *

import os

class DeOldify_Model(Model):
    # 模型基础信息定义
    name='deoldify'
    label='图片上色'
    describe="图片上色"
    field="机器视觉"
    scenes="图像合成"
    status='online'
    version='v20221001'
    pic='example.jpg'
    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待识别图片', describe='用于上色的黑白图片')
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "img_file_path": "test.png"
            }
        }
    ]

    # 加载模型
    def load_model(self):
        plt.style.use('dark_background')
        torch.backends.cudnn.benchmark = True
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
        self.colorizer = get_image_colorizer(root_folder=Path('/DeOldify'),artistic=True)

    # 推理
    @pysnooper.snoop()
    def inference(self,img_file_path):
        render_factor = 35
        save_path = self.colorizer.plot_transformed_image(path=img_file_path, render_factor=render_factor, compare=True)
        back=[{
            "image":str(save_path)
        }]
        return back


model=DeOldify_Model()
# model.load_model()
# result = model.inference(img_file_path='test.png')  # 测试
# print(result)

if __name__=='__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py inference --arg1 xx --arg2 xx
    # python app.py web
    model.run()