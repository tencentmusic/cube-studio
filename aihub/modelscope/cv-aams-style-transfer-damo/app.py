import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field


import pysnooper
import os

class CV_AAMS_STYLE_TRANSFER_DAMO_Model(Model):
    # 模型基础信息定义
    name='cv-aams-style-transfer-damo'   # 该名称与目录名必须一样，小写
    label='AAMS图像风格迁移'
    describe="给定内容图像和风格图像作为输入，风格迁移模型会自动地将内容图像的风格、纹理特征变换为风格图像的类型，同时保证图像的内容特征不变"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "11162"
    frameworks = "tensorflow"
    doc = "https://modelscope.cn/models/damo/cv_aams_style-transfer_damo/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='content', label='',describe='',default='',validators=None),
        Field(type=Field_type.image, name='style', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "",
            "input": {
                "content": "https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-style-transfer/style_transfer_content.jpg",
                "style": "https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-style-transfer/style_transfer_style.jpg"
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
        
        self.p = pipeline('image-style-transfer', 'damo/cv_aams_style-transfer_damo')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,content,style,**kwargs):
        result = self.p({'content':content,'style':style})
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

model=CV_AAMS_STYLE_TRANSFER_DAMO_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(content='https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-style-transfer/style_transfer_content.jpg',style='https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-style-transfer/style_transfer_style.jpg')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
