import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server

import pysnooper
import os

class CV_MOBILENET_FACE_2D_KEYPOINTS_ALIGNMENT_Model(Model):
    # 模型基础信息定义
    name='cv-mobilenet-face-2d-keypoints-alignment'   # 该名称与目录名必须一样，小写
    label='106点人脸关键点-通用领域-2D'
    describe="人脸2d关键点对齐模型"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "6123"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_mobilenet_face-2d-keypoints_alignment/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='arg0', label='带人脸的输入图像',describe='带人脸的输入图像',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "/root/.cache/modelscope/hub/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/1.jpg"
            }
        },
        {
            "label": "示例2",
            "input": {
                "arg0": "/root/.cache/modelscope/hub/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/2.jpg"
            }
        },
        {
            "label": "示例3",
            "input": {
                "arg0": "/root/.cache/modelscope/hub/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/3.jpg"
            }
        },
        {
            "label": "示例4",
            "input": {
                "arg0": "/root/.cache/modelscope/hub/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/4.jpg"
            }
        },
        {
            "label": "示例5",
            "input": {
                "arg0": "/root/.cache/modelscope/hub/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/5.jpg"
            }
        },
        {
            "label": "示例6",
            "input": {
                "arg0": "/root/.cache/modelscope/hub/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/6.jpg"
            }
        },
        {
            "label": "示例7",
            "input": {
                "arg0": "/root/.cache/modelscope/hub/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/7.jpg"
            }
        },
        {
            "label": "示例10",
            "input": {
                "arg0": "/root/.cache/modelscope/hub/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/10.jpg"
            }
        },
        {
            "label": "示例11",
            "input": {
                "arg0": "/root/.cache/modelscope/hub/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/11.jpg"
            }
        }
    ]

    # 训练的入口函数，将用户输入参数传递
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        pass


    # 加载模型
    def load_model(self,save_model_dir=None,**kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        self.p = pipeline('face-2d-keypoints', 'damo/cv_mobilenet_face-2d-keypoints_alignment')

    # 推理
    # @pysnooper.snoop()
    def inference(self,arg0,**kwargs):
        pass
        back=[
            {
                "image": result_img,
                "text": result_text,
                "video": result_video,
                "audio": result_audio,
                "markdown":result_markdown
            }
        ]
        return back

model=CV_MOBILENET_FACE_2D_KEYPOINTS_ALIGNMENT_Model()

# model.load_model()
# result = model.inference(arg1='测试输入文本',arg2='test.jpg')  # 测试
# print(result)

if __name__=='__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py inference --arg1 xx --arg2 xx
    # python app.py web --save_model_dir xx
    # python app.py download_model 用于再构建镜像下载一些预训练模型
    model.run()
