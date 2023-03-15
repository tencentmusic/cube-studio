import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_ECBSR_IMAGE_SUPER_RESOLUTION_MOBILE_Model(Model):
    # 模型基础信息定义
    name='cv-ecbsr-image-super-resolution-mobile'   # 该名称与目录名必须一样，小写
    label='ECBSR端上图像超分模型'
    describe="ECBSR模型基于Edgeoriented Convolution Block (ECB)模块构建，完整模型可导出为简洁的CNN网络结构，适用于移动端、嵌入式等严格限制算力的场景。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "228"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_ecbsr_image-super-resolution_mobile/summary"

    train_inputs = []

    inference_inputs = [
         Field(type=Field_type.image, name='arg0', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[]

    # 训练的入口函数，将用户输入参数传递
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        pass


    # 加载模型
    def load_model(self,save_model_dir=None,**kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        self.p = pipeline('image-super-resolution', 'damo/cv_ecbsr_image-super-resolution_mobile')

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

model=CV_ECBSR_IMAGE_SUPER_RESOLUTION_MOBILE_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference()  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
