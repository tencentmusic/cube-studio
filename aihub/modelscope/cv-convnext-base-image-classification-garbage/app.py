import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os
import pandas as pd

class CV_CONVNEXT_BASE_IMAGE_CLASSIFICATION_GARBAGE_Model(Model):
    # 模型基础信息定义
    name='cv-convnext-base-image-classification-garbage'   # 该名称与目录名必须一样，小写
    label='ConvNeXt图像分类-中文-垃圾分类'
    describe="自建265类常见的生活垃圾标签体系，15w张图片数据，包含可回收垃圾、厨余垃圾、有害垃圾、其他垃圾4个标准垃圾大类，覆盖常见的食品，厨房用品，家具，家电等生活垃圾，标签从海量中文互联网社区语料进行提取，整理出了频率较高的常见生活垃圾名称。模型结构采用ConvNeXt-Base结构, 经过大规模数据集ImageNet-22K预训练后，在数据集上进行微调。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "3069"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_convnext-base_image-classification_garbage/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='上传一张需要分类的垃圾的图片',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "image": "/mnt/workspace/.cache/modelscope/damo/cv_convnext-base_image-classification_garbage/resources/test.jpg"
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
        
        self.p = pipeline('image-classification', 'damo/cv_convnext-base_image-classification_garbage')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):

        result = pd.DataFrame(self.p(image))
        result['scores'] = result['scores'].apply(lambda x:' {:.4%}'.format(x))
        result = result[['labels','scores']]
        result.columns = [['分类','可能性']]

        back=[
            {
                "markdown": result.to_markdown()
            }
        ]
        return back




model=CV_CONVNEXT_BASE_IMAGE_CLASSIFICATION_GARBAGE_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference(image='/mnt/workspace/.cache/modelscope/damo/cv_convnext-base_image-classification_garbage/resources/test.jpg')  # 测试
# print(result)

# 测试后打开此部分
if __name__=='__main__':
    model.run()
