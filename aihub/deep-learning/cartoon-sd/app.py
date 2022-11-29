import io, sys, os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '/naifu')))

import random
from datetime import datetime
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server, Field, Field_type
import pysnooper
import os
import random
import re
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from hydra_node.config import init_config_model
from hydra_node.models import EmbedderModel
from typing import Optional, List
from typing_extensions import TypedDict
import socket
from hydra_node.sanitize import sanitize_input
import uvicorn
from typing import Union
import time
import gc
import signal
import base64
import traceback
import threading
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
import argparse

uc_dic = {
    'None': 'lowres',
    'Low Quality + Bad Anatomy': 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
    'Low Quality': 'lowres, text, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
    'NSFW + Low Quality + Bad Anatomy': 'nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
    'NSFW + Low Quality': 'nsfw, lowres, text, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
    'NSFW': 'nsfw, lowres'
}


def set_parameter(text='Pictures of astronauts on horseback', d_steps=30, samples=1, devi='cuda', se=None,
                  uc=uc_dic['NSFW + Low Quality + Bad Anatomy']):
    global uc_dic
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, nargs="?", default='masterpiece, best quality, ' + text, )
    parser.add_argument("--image", type=str, nargs="?", default=None)
    parser.add_argument("--module", type=str, default=None, )
    parser.add_argument("--masks", type=str, default=None, )
    parser.add_argument("--latent_channels", type=int, default=4, )
    parser.add_argument("--n_samples", type=int, default=samples)
    parser.add_argument("--steps", type=int, default=d_steps)
    parser.add_argument("--ddim_eta", type=float, default=0.0, )
    parser.add_argument("--n_iter", type=int, default=1, )
    parser.add_argument("--height", type=int, default=768, )
    parser.add_argument("--width", type=int, default=512, )
    parser.add_argument("--scale", type=float, default=12.0, )
    parser.add_argument("--strength", type=float, default=0.69, )
    parser.add_argument("--noise", type=float, default=0.667, )
    parser.add_argument("--dynamic_threshold", type=str, default=None)
    parser.add_argument("--downsampling_factor", type=int, default=8, )
    parser.add_argument("--grid_size", type=int, default=4, )
    parser.add_argument("--device", type=str, default=devi, )
    parser.add_argument("--seed", type=int, default=se, )
    parser.add_argument("--uc", type=str,
                        default=uc)
    parser.add_argument("--fixed_code", action="store_true", default=False)
    parser.add_argument("--advanced", action="store_true", default=False)
    parser.add_argument("--sampler", type=str, help="sampler",
                        choices=["ddim", "plms", "k_euler_ancestral"], default="k_euler_ancestral", )
    opt = parser.parse_args()
    return opt


class Cartoon_SD_Model(Model):
    # 模型基础信息定义
    name = 'cartoon-sd'
    label = '文字转图像-动漫版'
    describe = "输入一串文字描述，可生成相应的动漫图片，描述越详细越好哦~"
    field = "机器视觉"
    scenes = "图像创作"
    status = 'online'
    version = 'v20221125'
    doc = 'https://github.com/tencentmusic/cube-studio'  # 'https://帮助文档的链接地址'
    pic = 'example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址

    inference_resource = {
        "resource_gpu": "1"
    }

    inference_inputs = [
        Field(type=Field_type.text, name='prompt', label='输入的文字内容',
              describe='输入的文字内容，描述越详细越好', default='a photograph of an astronaut riding a horse'),
        Field(type=Field_type.text_select, name='style', label='图像效果参数', default='Low Quality + Bad Anatomy',
              choices=['None', 'Low Quality + Bad Anatomy', 'Low Quality', 'NSFW + Low Quality + Bad Anatomy',
                       'NSFW + Low Quality', 'NSFW'], describe='效果参数'),
        Field(type=Field_type.text_select, name='n_samples', label='推理出的图像数量',
              describe='结果中所展示的图片数量，数量越多则会导致性能下降', default="1",
              choices=[str(x + 1) for x in range(20)]),

    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "prompt": 'a photograph of an astronaut riding a horse',
                "n_samples": 1,
                "style": 'None'
            }
        }
    ]

    # 加载模型
    # @pysnooper.snoop()
    def load_model(self):
        self.model, config, model_hash = init_config_model()
        try:
            embedmodel = EmbedderModel()
        except Exception as e:
            print("couldn't load embed model, suggestions won't work:", e)
            embedmodel = False
        logger = config.logger
        try:
            config.mainpid = int(open("gunicorn.pid", "r").read())
        except FileNotFoundError:
            config.mainpid = os.getpid()
        mainpid = config.mainpid
        hostname = socket.gethostname()
        sent_first_message = False

    def do_job(self, optim):
        try:
            if optim.seed is None:
                optim.seed = random.randint(0, 100000000)
            images = self.model.sample(optim)
            ndarray_convert_img_list = []
            for image in images:
                ndarray_convert_img_list.append(Image.fromarray(image))
            return True, optim.seed, ndarray_convert_img_list
        except Exception as ex:
            print(ex)
            return False, 0, ''

    # 推理
    @pysnooper.snoop()
    def inference(self, prompt, style, n_samples=1):
        global uc_dic
        try:
            n_samples = int(n_samples)
            time_str = datetime.now().strftime('%Y%m%d%H%M%S')
            opt = set_parameter()
            opt.prompt = prompt
            opt.n_samples = n_samples
            opt.seed = random.randint(0, 100000000)
            opt.uc = uc_dic[style]
            x, y, z = self.do_job(opt)
            os.makedirs('result', exist_ok=True)
            save_index = 0
            re_list = []
            if x:
                for one in z:
                    one.save(f'result/{time_str}-{y}-{save_index}.png')
                    re_list.append(f'result/{time_str}-{y}-{save_index}.png')
                    save_index += 1
            back = [
                {
                    "image": img_path
                } for img_path in re_list
            ]
            return back
        except Exception as ex:
            print(ex)
            back = [{
                "text": f'出现错误，请联系开发人处理{str(ex)}'
            }]
            return back


model = Cartoon_SD_Model()
# model.load_model()
# result = model.inference(prompt='a photograph of an astronaut riding a horse')  # 测试
# print(result)

# 启动服务
server = Server(model=model)
server.server(port=8080)
