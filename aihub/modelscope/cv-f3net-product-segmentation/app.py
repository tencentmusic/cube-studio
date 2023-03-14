import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_F3NET_PRODUCT_SEGMENTATION_Model(Model):
    # 模型基础信息定义
    name='cv-f3net-product-segmentation'   # 该名称与目录名必须一样，小写
    label='图像分割-商品展示图场景的商品分割-电商领域'
    describe="通用商品分割模型，适用于商品展示图场景"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "38301"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_F3Net_product-segmentation/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='input_path', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "input_path": "/mnt/workspace/.cache/modelscope/damo/cv_F3Net_product-segmentation/test_segmentation.jpg"
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
        
        self.p = pipeline('product-segmentation', 'damo/cv_F3Net_product-segmentation')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,input_path,**kwargs):
        result =self.p({'input_path':input_path})
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

model=CV_F3NET_PRODUCT_SEGMENTATION_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(input_path='/mnt/workspace/.cache/modelscope/damo/cv_F3Net_product-segmentation/test_segmentation.jpg')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
