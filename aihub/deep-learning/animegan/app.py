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
    describe="图片的全新动漫风格化，宫崎骏或新海诚风格的动漫，以及4种关于人脸的风格转换。"
    field="机器视觉"
    scenes="图像合成"
    status='online'
    version='v20221001'
    doc='https://github.com/TachibanaYoshino/AnimeGANv3' # 'https://帮助文档的链接地址'
    pic='example.png'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    inference_inputs = [
        Field(type=Field_type.image, name='img_path', label='源图片',describe='待风格化的原始图片'),
        Field(type=Field_type.text_select, name='style', label='风格',default='宫崎骏 风格', choices=['宫崎骏 风格','Arcane 风格','新海诚 风格','肖像素描 风格','川普 风格'], describe='风格类型',validators=Validator(max=1))
    ]
    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples = [
        {
            "lable": "宫崎骏 风格",
            "input": {
                "img_path": 'jp_Hayao.jpg',
                "style": "宫崎骏 风格"
            },
        },
        {
            "lable": "Arcane 风格",
            "input": {
                "img_path": 'jp_Arcane.jpg',
                "style": "Arcane 风格"
            },
        },
        {
            "lable": "新海诚 风格",
            "input": {
                "img_path": 'jp_Shinkai.jpg',
                "style": "新海诚 风格"
            },
        },
        {
            "lable": "肖像素描 风格",
            "input": {
                "img_path": 'jp_PortraitSketch.jpg',
                "style": "肖像素描 风格"
            },
        },
        {
            "lable": "川普 风格",
            "input": {
                "img_path": 'jp_Trump.jpg',
                "style": "川普 风格"
            },
        }
    ]

    # 加载模型
    def load_model(self):
        pass

    # 推理
    @pysnooper.snoop()
    def inference(self,img_path, style='宫崎骏', if_face=None):
        print(img_path, style, if_face)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if 'Arcane' in style:
            f = "A"
        elif "川普" in style:
            f = "T"
        elif "新海诚" in style:
            f = "S"
        elif "肖像素描" in style:
            f = "P"
        elif "宫崎骏" in style:
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
# model.load_model()
# result = model.inference(img_path='jp_Hayao.jpg',style='宫崎骏')  # 测试
# result = model.inference(img_path='jp_Arcane.jpg',style='Arcane')  # 测试
# result = model.inference(img_path='jp_Shinkai.jpg',style='新海诚')  # 测试
# result = model.inference(img_path='jp_PortraitSketch.jpg',style='肖像素描')  # 测试
# result = model.inference(img_path='jp_Trump.jpg',style='川普')  # 测试

# # 启动服务
server = Server(model=model)
server.server(port=8080)

