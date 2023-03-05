import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type,Validator

import pysnooper
import os

from web.aihub import start

class Face_paint_Model(Model):
    # 模型基础信息定义
    name='face-paint'   # 该名称与目录名必须一样，小写
    label='ai国庆头像生成'
    describe="上传个人照片，自动生成AI头像"
    field="机器视觉"
    scenes="图像合成"
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址

    inference_inputs = [
        Field(type=Field_type.text, name='name', label='您的姓氏',
              describe='将会被合成到图片中', default='姓'),
        Field(type=Field_type.image, name='face_image', label='个人照片上半身',
              describe='会自动进行人体分割，上半身截取，人体卡通化，照片合成',default='')
    ]

    # 加载模型
    def load_model(self,model_dir=None,**kwargs):
        # self.model = load("/xxx/xx/a.pth")
        pass

    # 推理
    # @pysnooper.snoop()
    def inference(self,name,face_image,**kwargs):
        all_back_image = start(
            img_path=face_image,
            text=name
        )
        result_img='result.jpg'
        result_text='cat,dog'
        result_video='https://pengluan-76009.sz.gfp.tencent-cloud.com/cube-studio%20install.mp4'
        result_audio = 'test.wav'
        result_markdown=open('test.md',mode='r').read()
        back=[
            {
                "image":result_img,
                "text":result_text,
                "video":result_video,
                "audio":result_audio,
                "html":'<a href="http://www.data-master.net/frontend/aihub/model_market/model_all">查看全部</a>',
                "markdown":result_markdown
            },
            {
                "image": result_img,
                "text": result_text,
                "video": result_video,
                "audio": result_audio,
                "markdown":result_markdown
            }
        ]
        return back

model=Face_paint_Model()
# model.load_model()
# result = model.inference(arg1='测试输入文本',arg2='test.jpg')  # 测试
# print(result)

if __name__=='__main__':
    # # 启动服务
    server = Server(model=model)
    server.server(port=8080)

