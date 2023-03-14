import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_CSRNET_IMAGE_COLOR_ENHANCE_MODELS_Model(Model):
    # 模型基础信息定义
    name='cv-csrnet-image-color-enhance-models'   # 该名称与目录名必须一样，小写
    label='CSRNet图像调色'
    describe="基于CSRNet实现的图像色彩增强算法，输入待增强图像，输出色彩增强后的图像。CSRNet通过计算全局调整参数并将之作用于条件网络得到的特征，保证效果的基础之上实现轻便高效的训练和推理。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "12048"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_csrnet_image-color-enhance-models/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='arg0', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_csrnet_image-color-enhance-models/data/1.png"
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
        
        self.p = pipeline('image-color-enhancement', 'damo/cv_csrnet_image-color-enhance-models')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)
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

model=CV_CSRNET_IMAGE_COLOR_ENHANCE_MODELS_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(arg0='/mnt/workspace/.cache/modelscope/damo/cv_csrnet_image-color-enhance-models/data/1.png')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
