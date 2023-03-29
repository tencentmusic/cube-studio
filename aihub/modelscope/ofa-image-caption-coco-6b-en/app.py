import base64
import io,sys,os
import numpy as np
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class OFA_IMAGE_CAPTION_COCO_6B_EN_Model(Model):
    # 模型基础信息定义
    name='ofa-image-caption-coco-6b-en'   # 该名称与目录名必须一样，小写
    label='OFA图像描述-英文-通用领域-6B'
    describe="根据用户输入的任意图片，AI智能创作模型写出“一句话描述”，可用于图像标签和图像简介。"
    field="多模态"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "145"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/ofa_image-caption_coco_6b_en/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='image', label='图片',describe='图片',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "image": "https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg"
            }
        },
        {
            "label": "示例2",
            "input": {
                "image": "https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/vqa2.jpg"
            }
        },
        {
            "label": "示例3",
            "input": {
                "image": "https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/vqa1.jpg"
            }
        },
        {
            "label": "示例4",
            "input": {
                "image": "https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/vqa3.jpg"
            }
        },
        {
            "label": "示例5",
            "input": {
                "image": "https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/vqa4.jpeg"
            }
        }
    ]

    # 训练的入口函数，此函数会自动对接pipeline，将用户在web界面填写的参数传递给该方法
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        pass
        # 训练的逻辑
        # 将模型保存到save_model_dir 指定的目录下


    # 加载模型，所有一次性的初始化工作可以放到该方法下。注意save_model_dir必须和训练函数导出的模型结构对应
    def load_model(self,save_model_dir=None,**kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        self.p = pipeline('image-captioning', 'damo/ofa_image-caption_coco_6b_en')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:np.ndarray,**kwargs)->np.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):
        result = self.p(image)
        result = result['caption']
        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back=[
            {
                "text": str(result)
            }
        ]
        return back

model=OFA_IMAGE_CAPTION_COCO_6B_EN_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(image='https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg')  # 测试
# print(result)

# 模型启动web时使用
if __name__=='__main__':
    model.run()

'''
Date: 2023-03-26
Tested by: 秋水泡茶
模型大小：22.4G
模型效果：MS COCO Caption训练的通用根据图片生成文本算法模型，模型文件较大，容器启动较慢，占用显存较高。
推理性能：以1920*1080分辨率图像为例，平均推理时间在20s左右。
测试环境：CPU：Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz, GPU：Tesla V100 32G
占用GPU显存：正常运行状态下，占用显存26145M，每次运行单卡功耗160W左右，单卡占用率70%左右。
巧妙使用方法：
'''