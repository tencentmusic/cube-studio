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
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "555"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_daflow_virtual-try-on_base/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='masked_model', label='模特图',describe='模特图，',default='https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_model.jpg',validators=None),
        Field(type=Field_type.image, name='pose', label='骨架图',describe='图片',default='https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_pose.jpg',validators=None),
        Field(type=Field_type.image, name='cloth', label='衣服平铺图',describe='图片',default='https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_cloth.jpg',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "masked_model": "https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_model.jpg",
                "pose": "https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_pose.jpg",
                "cloth": "https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_cloth.jpg"
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
    def inference(self,masked_model,pose,cloth,**kwargs):
        import cv2,time
        from modelscope.outputs import OutputKeys
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        input_imgs = {
            'masked_model': masked_model,
            'pose': pose,
            'cloth': cloth
        }
        img = self.p(input_imgs)[OutputKeys.OUTPUT_IMG]

        savePath = 'result/result_' + str(int(1000 * time.time())) + '.jpg'
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        if os.path.exists(savePath):
            os.remove(savePath)

        cv2.imwrite(savePath, img[:, :, ::-1])
        
        back=[
            {
                "image": savePath
            }
        ]
        return back

model=CV_DAFLOW_VIRTUAL_TRY_ON_BASE_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference(masked_model='https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_model.jpg',pose='https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_pose.jpg',cloth='https://m6-public.oss-cn-hangzhou.aliyuncs.com/demo/virtual_tryon_cloth.jpg')  # 测试
# print(result)

# 测试后打开此部分
if __name__=='__main__':
    model.run()

# 模型大小160M
# 模型输入太限定，需要结合其他模型一起使用，比如姿态识别，关键点识别
# 而且图片非常容易出错，
# v100 gpu推理 耗时1.2s