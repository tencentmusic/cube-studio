import base64
import io,sys,os

from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type
import pysnooper
import json

import io
import requests
from PIL import Image
import torch
import numpy

from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id

import os

class Panoptic_Model(Model):
    # 模型基础信息定义
    name='panoptic'
    label='图片识别'
    description="resnet50 图像识别"
    field="机器视觉"
    scenes="目标识别"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/DeOldify'
    pic='test.png'

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待目标识别图片', describe='用于目标识别的原始图片')
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "img_file_path": "test.jpg"
            }
        }
    ]

    # 加载模型
    def load_model(self):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
        self.model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
        self.config = json.load(open('config.json'))

    # 推理
    @pysnooper.snoop()
    def inference(self,img_file_path):
        image = Image.open(img_file_path)
        # prepare image for the model
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # forward pass
        outputs = self.model(**inputs)

        # use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = self.feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

        # # the segmentation is stored in a special-format png
        # panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        # panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
        # # retrieve the ids corresponding to each mask
        # panoptic_seg_id = rgb_to_id(panoptic_seg)
        #
        # print(panoptic_seg_id)

        cates = [seg['category_id'] for seg in result['segments_info']]
        class_type = [self.config['id2label'][str(cate_id)] for cate_id in cates]
        back = [
            {
                "text": text
            } for text in class_type
        ]
        return back

model=Panoptic_Model()
# model.load_model()
# result = model.inference(img_file_path='test.jpg')  # 测试
# print(result)

# # 启动服务
server = Server(model=model)
server.server(port=8080)

