import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_DAFLOW_VIRTUAL_TRY_ON_BASE_Model(Model):
    # 模型基础信息定义
    name='cv-daflow-virtual-try-on-base'   # 该名称与目录名必须一样，小写
    label='DAFlow虚拟试衣模型-VITON数据'
    describe="DAFlow是一种单阶段虚拟试衣框架，无需中间分割结果作为label，直接用模特上身图作为监督。同时本工作提出一种新的空间变换结构，在虚拟试衣和一些变换任务上达到SOTA."
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "555"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_daflow_virtual-try-on_base/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='Image1', label='图片',describe='图片',default='',validators=None),
        Field(type=Field_type.image, name='Image2', label='图片',describe='图片',default='',validators=None),
        Field(type=Field_type.image, name='Image3', label='图片',describe='图片',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "Image1": "https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_model.jpg",
                "Image2": "https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_pose.jpg",
                "Image3": "https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_cloth.jpg"
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
        
        self.p = pipeline('virtual-try-on', 'damo/cv_daflow_virtual-try-on_base')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,Image1,Image2,Image3,**kwargs):
        input_imgs = {
            'masked_model': Image1,
            'pose': Image2,
            'cloth': Image3
        }
        img = self.p(input_imgs)[OutputKeys.OUTPUT_IMG]
        cv2.imwrite('demo.jpg', img[:, :, ::-1])
        
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

model=CV_DAFLOW_VIRTUAL_TRY_ON_BASE_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(Image1='https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_model.jpg',Image2='https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_pose.jpg',Image3='https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_cloth.jpg')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
