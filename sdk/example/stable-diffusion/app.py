import base64
import datetime
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type,Validator

import pysnooper
import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


class SD_Model(Model):
    # 模型基础信息定义
    name='stable-diffusers'
    label='图片生成艺术图'
    description="输入一串文字描述，可生成相应的图片"
    field="机器视觉"
    scenes="图像合成"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub' # 'https://帮助文档的链接地址'
    pic='https://images.nightcafe.studio//assets/stable-tile.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.text, name='image_describe', label='文字描述',
              describe='根据文字描述生成图片',default='')
    ]

    # 加载模型
    def load_model(self):
        model_id = "CompVis/stable-diffusion-v1-4"
        self.device = 'cpu'  #cuda

        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
        self.pipe = self.pipe.to(self.device)

    # 推理
    @pysnooper.snoop()
    def inference(self,image_describe):

        with autocast(self.device):
            image = self.pipe(image_describe, guidance_scale=7.5).images[0]
        save_path = 'result/%s.jpg'%datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        image.save(save_path)

        back=[
            {
                "image":save_path,
                "text":image_describe
            }
        ]
        return back

model=SD_Model(init_shell=False)
model.load_model()
result = model.inference(image_describe = "a photo of an astronaut riding a horse on mars")  # 测试
print(result)

# # 启动服务
server = Server(model=model)
server.web_examples.append({
    "image_describe":'a photo of an astronaut riding a horse on mars'
})
server.server(port=8080)

