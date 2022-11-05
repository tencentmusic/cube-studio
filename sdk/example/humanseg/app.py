import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type,Validator

import pysnooper
import os


import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

from paddleseg.utils import get_sys_env, logger, get_image_list
from start import seg_image,seg_video,parse_args
from infer import Predictor


class HumanSeg_Model(Model):
    # 模型基础信息定义
    name='humanseg'
    label='人体分割背景替换'
    description="人体分割背景替换，视频会议背景替换"
    field="机器视觉"
    scenes="图像合成"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub' # 'https://帮助文档的链接地址'
    pic='https://github.com/juncaipeng/raw_data/blob/master/images/portrait_bg_replace_1.gif?raw=true'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址

    inference_inputs = [
        Field(type=Field_type.video, name='human_video', label='人体视频'),
        Field(type=Field_type.image, name='background', label='要替换的背景图片',validators=Validator(required=True)),

    ]

    # 加载模型
    def load_model(self):
        args = parse_args()
        env_info = get_sys_env()
        args.use_gpu = True if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else False
        args.config='/inference_models/portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax/deploy.yaml'
        args.use_post_process=True
        self.args=args
        self.predictor = Predictor(args)


    # 推理
    @pysnooper.snoop()
    def inference(self,human_video,background):
        file_name = os.path.basename(human_video).split('.')[0]
        self.args.save_dir = 'result/%s.avi'%file_name
        os.makedirs(os.path.dirname(self.args.save_dir),exist_ok=True)

        self.args.video_path=human_video
        self.args.bg_img_path=background
        seg_video(self.args,self.predictor)
        back=[
            {
                "video":self.args.save_dir
            }
        ]
        return back

model=HumanSeg_Model()
model.load_model()
result = model.inference(background='/data/images/bg_2.jpg',human_video='/data/videos/video_heng.mp4')  # 测试
print(result)

# # 启动服务
server = Server(model=model)
server.web_examples.append({
    "background":'/data/images/bg_2.jpg',
    "human_video":'/data/videos/video_heng.mp4',
})
server.server(port=8080)

