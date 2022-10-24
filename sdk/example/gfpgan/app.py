import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type
import pysnooper
import os


import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from gfpgan import GFPGANer

class GFPGAN_Model(Model):
    # 模型基础信息定义
    name='gfpgan'
    label='图片修复'
    description="低分辨率照片修复，清晰度增强"
    field="机器视觉"
    scenes="图像合成"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/face-repair'
    pic='https://p6.toutiaoimg.com/origin/tos-cn-i-qvj2lq49k0/6a284d35f42b414d9f4dcb474b0e644f'
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待修复图片', describe='需要进行修复的低清晰图片')
    ]

    # 加载模型
    @pysnooper.snoop()
    def load_model(self):
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
        arch = 'clean'
        channel_multiplier = 2

        model_path = '/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth'

        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)

    # 推理
    @pysnooper.snoop()
    def inference(self,img_file_path):
        input_img = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5)

        # save restored img
        save_restore_path=''
        if restored_img is not None:
            os.makedirs('result', exist_ok=True)
            save_restore_path = os.path.join('result', os.path.basename(img_file_path))
            imwrite(restored_img, save_restore_path)

        back=[{
            "image":save_restore_path
        }]
        return back

model=GFPGAN_Model()
model.load_model()
result = model.inference(img_file_path='test.png')  # 测试
print(result)

# # # 启动服务
server = Server(model=model)
server.web_examples.append(
    {"img_file_path":"test.png"}
)
server.server(port=8080)

