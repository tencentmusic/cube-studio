import json,io
from PIL import Image
import torch
import gradio as gr
import os,logging
import datetime
import pysnooper
import requests
import re,shutil
import uvicorn
import copy,time
from models.experimental import attempt_load
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from gradio.components import Component
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.getenv('MODELPATH','/yolov7/weights/yolov7.pt')
device='cpu'

# 加载模型
# 这里是添加的gpu识别
resource_gpu = os.getenv('RESOURCE_GPU', '')
resource_gpu = resource_gpu.split('(')[0]
resource_gpu = int(float(resource_gpu)) if resource_gpu else 0
if resource_gpu:
    resource_gpu = list(range(resource_gpu))
    resource_gpu = [str(x) for x in resource_gpu]
    device = ','.join(resource_gpu)
# 上面是添加的gpu识别
device = select_device(device)

# Load model
model = attempt_load(model_path, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride


# @pysnooper.snoop()
def inference(source):
    imgsz = 640
    if 'http://' in source or 'https://' in source:
        response = requests.get(source)
        ext = source[source.rindex(".") + 1:]
        ext = ext if len(ext) < 6 else 'jpg'
        source = f'input-{int(time.time() * 1000)}.{ext}'

        if os.path.exists(source):
            os.remove(source)


        # 确保请求成功
        if response.status_code == 200:
            # 将视频内容写入本地文件
            with open(source, "wb") as file:
                file.write(response.content)
                print(f"文件已成功保存到: {source}")
        else:
            print(f"请求失败，状态码: {response.status_code}")


    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        print(pred)
        save_dir = 'result'
        os.makedirs(save_dir,exist_ok=True)
        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            save_path = os.path.join(save_dir,f'{int(time.time() * 1000)}.jpg')  # img.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)


            # Save results (image with detections)

            cv2.imwrite(save_path, im0)
            print(f" The image with the result is saved in: {save_path}")
            return save_path



label = 'cube-studio开源平台yolov7目标识别推理服务'
describe = 'cube studio开源云原生一站式机器学习/深度学习AI平台，支持sso登录，多租户/多项目组，数据资产对接，notebook在线开发，拖拉拽任务流pipeline编排，多机多卡分布式算法训练，超参搜索，推理服务VGPU，多集群调度，边缘计算，serverless，标注平台，自动化标注，数据集管理，大模型一键微调，llmops，私有知识库，AI应用商店，支持模型一键开发/推理/微调，私有化部署，支持国产cpu/gpu/npu芯片，支持RDMA，支持pytorch/tf/mxnet/deepspeed/paddle/colossalai/horovod/spark/ray/volcano分布式'
gradio_examples=[
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000000597.jpg",
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000000797.jpg",
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000000897.jpg",
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000001397.jpg",
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000001497.jpg",
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000001697.jpg"
]
with gr.Blocks(title=label,theme=gr.themes.Default(text_size='lg')) as demo:

    with gr.Row():
        html=f'<h1 style="text-align: center; margin-bottom: 1rem">{label}</h1>'
        title = gr.HTML(value = html)
    # 介绍
    with gr.Row():
        description = gr.Markdown(value = describe)
    with gr.Row():
        with gr.Tab("推理"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 输入
                    inputs = gr.Image(value=None, label='待推理图片',type="filepath")
                    submit_button = gr.Button("提交")
                with gr.Column(scale=1):
                    outputs = gr.Image(label='图片输出结果')
                submit_button.click(inference, inputs=inputs,outputs=outputs)
            with gr.Row():
                # 遍历多个示例
                gr.Examples(
                    examples=gradio_examples,
                    inputs=inputs,
                    outputs=outputs,
                    fn=inference,
                    cache_examples=False,
                )
            # with gr.Row():
            #     path = os.path.join(current_dir,'gradio_rec.txt')
            #     if os.path.exists(path):
            #         choices = open(path).readlines()
            #         choices = [x.strip() for x in choices if x.strip()]
            #         gr.Gallery(value=choices,label='其他应用',show_label=False,allow_preview=False)
        # with gr.Tab("接口"):
        #     api_markdown = ''
        #     if os.path.exists(os.path.join(current_dir, 'yolov7_api.md')):
        #         api_markdown = open(os.path.join(current_dir, 'yolov7_api.md')).read().strip()
        #     jiekou = gr.Markdown(value=api_markdown)

demo.launch(server_name='0.0.0.0',server_port=8080)

#
# from fastapi import FastAPI, Request
# import base64
#
# from PIL import Image
# app = FastAPI()
#
# # define your API endpoint
# @app.post('/v1/models/yolov7/versions/20240101/predict')
# async def predict(request: Request):
#     data = await request.json()
#     image_data = data['image']
#     # 解码图像
#     image_bytes = base64.b64decode(image_data)
#     image = Image.open(io.BytesIO(image_bytes))
#     save_path = f'input/{int(time.time())}.jpg'
#     os.makedirs('input',exist_ok=True)
#     image.save(save_path)
#     # 推理预测
#     result = inference(save_path)
#     logging.info('结果图片地址'+result)
#     # 返回结果
#     img = Image.open(result)
#     # 将图片转换为字节流
#     buf = io.BytesIO()
#     img.save(buf, format='JPEG')
#
#     # 获取字节流的二进制内容
#     byte_data = buf.getvalue()
#     # 将二进制内容编码为base64
#     base64_str = base64.b64encode(byte_data)
#
#     return {"image": base64_str.decode('utf-8')}
#
# app = gr.mount_gradio_app(app, demo, path=f"/gradio")
# uvicorn.run(app, host="0.0.0.0", port=8080)
# # uvicorn server:app --host '0.0.0.0' --port 8000 --reload








