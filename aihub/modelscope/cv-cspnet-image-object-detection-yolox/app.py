import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_CSPNET_IMAGE_OBJECT_DETECTION_YOLOX_Model(Model):
    # 模型基础信息定义
    name='cv-cspnet-image-object-detection-yolox'   # 该名称与目录名必须一样，小写
    label='实时目标检测-通用领域'
    describe="基于yolox小模型的通用检测模型"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "6899"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_cspnet_image-object-detection_yolox/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "image": "/mnt/workspace/.cache/modelscope/damo/cv_cspnet_image-object-detection_yolox/test_object_0.jpeg"
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
        
        self.p = pipeline('image-object-detection', 'damo/cv_cspnet_image-object-detection_yolox')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):
        result = self.p(image)
        back=[
            {
                "image": 'result/aa.jpg',
                "text": '结果文本',
                "video": 'result/aa.mp4',
                "audio": 'result/aa.mp3',
                "markdown":''
            }
        ]
        return back

model=CV_CSPNET_IMAGE_OBJECT_DETECTION_YOLOX_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(image='/mnt/workspace/.cache/modelscope/damo/cv_cspnet_image-object-detection_yolox/test_object_0.jpeg')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
