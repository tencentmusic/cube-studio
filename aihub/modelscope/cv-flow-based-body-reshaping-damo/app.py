import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_FLOW_BASED_BODY_RESHAPING_DAMO_Model(Model):
    # 模型基础信息定义
    name='cv-flow-based-body-reshaping-damo'   # 该名称与目录名必须一样，小写
    label='FBBR人体美型'
    describe="给定一张单个人物图像（半身或全身），无需任何额外输入，人体美型模型能够端到端地实现对人物身体区域（肩部，腰部，腿部等）的自动化美型处理。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "7195"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_flow-based-body-reshaping_damo/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_body_reshaping.jpg"
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
        
        self.p = pipeline('image-body-reshaping', 'damo/cv_flow-based-body-reshaping_damo')

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

model=CV_FLOW_BASED_BODY_RESHAPING_DAMO_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(image='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_body_reshaping.jpg')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
