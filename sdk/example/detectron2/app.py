import base64
import io,sys,os

from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type
import cv2
import pysnooper
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from PIL import ImageGrab, Image
from PIL import Image,ImageFont
from PIL import ImageDraw
import numpy

import os

class Detectron2_Model(Model):
    # 模型基础信息定义
    name='detectron2'
    label='视频换背景'
    description="行人图片分割，添加背景视频，合并为新背景人物视频"
    field="机器视觉"
    scenes="目标分割"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/detectron2'
    pic='https://user-images.githubusercontent.com/114121827/191642346-11715440-6c90-4709-ab21-8680034308cc.gif'
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.video, name='video_human_path', label='人物视频', describe='用于人物识别的行动视频'),
        Field(type=Field_type.video, name='video_background_path', label='要替换为的背景视频', describe='用于合成为背景的视频')
    ]

    # 加载模型
    def load_model(self):
        self.cfg = get_cfg()
        model_cfg_file = model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        self.cfg.merge_from_file(model_cfg_file)

        model_weight_url = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        self.cfg.MODEL.WEIGHTS = model_weight_url

        self.predictor = DefaultPredictor(self.cfg)

    # 推理
    @pysnooper.snoop()
    def inference(self,img_file_path):
        img = cv2.imread(img_file_path)
        out = self.predictor(img)
        res = self.detector.detect(img_file_path)

        back=[{
            "image":out_image_path,
            "text":res
        }]
        return back

model=Detectron2_Model(init_shell=False)
model.load_model()
result = model.inference(img_file_path='test.png')  # 测试
print(result)

# # 启动服务
server = Server(model=model)
server.web_examples.append(
    {"img_file_path":"test.jpg"}
)
server.server(port=8080)

