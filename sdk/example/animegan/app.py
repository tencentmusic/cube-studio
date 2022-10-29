import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type,Validator

import pysnooper
import os
import os
import cv2
import gradio as gr
import AnimeGANv3_src



class AnimeGANv3_Model(Model):
    # 模型基础信息定义
    name='animegan'
    label='动漫风格化'
    description="图片的全新动漫风格化，宫崎骏或新海诚风格的动漫，以及4种关于人脸的风格转换。"
    field="机器视觉"
    scenes="图像合成"
    status='online'
    version='v20221001'
    doc='https://github.com/TachibanaYoshino/AnimeGANv2' # 'https://帮助文档的链接地址'
    pic='https://raw.githubusercontent.com/TachibanaYoshino/AnimeGANv2/master/results/Hayao/concat/AE86.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.image, name='img_path', label='源图片',describe='待风格化的原始图片')
    ]

    # 加载模型
    def load_model(self):
        pass

    # 推理
    @pysnooper.snoop()
    def inference(self,img_path, Style='AnimeGANv3_Hayao', if_face=None):
        print(img_path, Style, if_face)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if Style == "AnimeGANv3_Arcane":
            f = "A"
        elif Style == "AnimeGANv3_Trump":
            f = "T"
        elif Style == "AnimeGANv3_Shinkai":
            f = "S"
        elif Style == "AnimeGANv3_PortraitSketch":
            f = "P"
        elif Style == "AnimeGANv3_Hayao":
            f = "H"
        else:
            f = "U"
        os.makedirs('result',exist_ok=True)
        save_path = os.path.join('result', os.path.basename(img_path))
        try:
            det_face = True if if_face == "Yes" else False
            output = AnimeGANv3_src.Convert(img, f, det_face)
            cv2.imwrite(save_path, output[:, :, ::-1])
        except RuntimeError as error:
            print('Error', error)

        back=[
            {
                "image":save_path
            }
        ]
        return back

model=AnimeGANv3_Model()
model.load_model()
result = model.inference(img_path='jp_38.jpg')  # 测试
result = model.inference(img_path='jp_13.jpg')  # 测试
result = model.inference(img_path='jp_20.jpg')  # 测试
# # # 启动服务
# server = Server(model=model)
# server.web_examples.append({
#     "img_path":'jp_38.jpg'
# })
# server.server(port=8080)
#
