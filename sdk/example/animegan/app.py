import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type,Validator

import pysnooper
import os

class AnimeGAN_Model(Model):
    # 模型基础信息定义
    name='animegan'
    label='动漫风格化'
    description="图片的全新动漫风格化，宫崎骏"
    field="机器视觉"
    scenes="图像合成"
    status='online'
    version='v20221001'
    doc='https://github.com/TachibanaYoshino/AnimeGANv2' # 'https://帮助文档的链接地址'
    pic='https://raw.githubusercontent.com/TachibanaYoshino/AnimeGANv2/master/results/Hayao/concat/AE86.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.image, name='img_path', label='源图片',
              describe='待风格化的原始图片')
    ]

    # 加载模型
    def load_model(self):
        # self.model = load("/xxx/xx/a.pth")
        pass

    # 推理
    @pysnooper.snoop()
    def inference(self,arg1,arg2=None,arg3=None,arg4=None,arg5=None,arg6=None,arg7=None,**kwargs):
        # save_path = os.path.join('result', os.path.basename(arg1))
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_img='result_img.jpg'
        result_text='cat,dog'
        result_video='https://pengluan-76009.sz.gfp.tencent-cloud.com/cube-studio%20install.mp4'
        result_audio = 'test.wav'
        back=[
            {
                "image":result_img,
                "text":result_text,
                "video":result_video,
                "audio":result_audio
            },
            {
                "image": result_img,
                "text": result_text,
                "video": result_video,
                "audio": result_audio
            }
        ]
        return back

model=AnimeGAN_Model()
model.load_model()
# result = model.inference(arg1='测试输入文本',arg2='test.jpg')  # 测试
# print(result)

# # 启动服务
server = Server(model=model)
server.web_examples.append({
    "arg1":'测试输入文本',
    "arg2":'test.jpg',
    "arg3": 'https://pengluan-76009.sz.gfp.tencent-cloud.com/cube-studio%20install.mp4'
})
server.server(port=8080)

