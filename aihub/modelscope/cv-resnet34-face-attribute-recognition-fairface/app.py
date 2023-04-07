import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import numpy
import pysnooper
import os
import cv2
import json
class CV_RESNET34_FACE_ATTRIBUTE_RECOGNITION_FAIRFACE_Model(Model):
    # 模型基础信息定义
    name='cv-resnet34-face-attribute-recognition-fairface'   # 该名称与目录名必须一样，小写
    label='人脸属性识别模型FairFace'
    describe="给定一张带人脸的图片，返回其性别和年龄范围。"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.png'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "1839"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_resnet34_face-attribute-recognition_fairface/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例0",
            "input": {
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/mog_face_detection.jpg"
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
        
        self.p = pipeline('face-attribute-recognition', 'damo/cv_resnet34_face-attribute-recognition_fairface')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):


        # result = self.p(image)
        img = cv2.imread(image)
        height, width, _ = img.shape
        if height > width:
            new_height = 1024
            new_width = int(width * new_height / height)
        else:
            new_width = 1024
            new_height = int(height * new_width / width)
        resized_img = cv2.resize(img, (new_width, new_height))
        result = self.p(resized_img)

        index_gender = result["scores"][0].index(max(result["scores"][0]))
        index_age = result["scores"][1].index(max(result["scores"][1]))

        # data = {
        #     "gender": result['labels'][0][index_gender],
        #     "age_range ": result['labels'][1][index_age]
        # }
        # json_str = json.dumps(data)
        res = '性别: %s, 年龄范围: %s'%(result['labels'][0][index_gender], result['labels'][1][index_age])
        back=[
            {
                "text": res
            }
        ]
        return back

model=CV_RESNET34_FACE_ATTRIBUTE_RECOGNITION_FAIRFACE_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(image='test-james.jpg')  # 测试
# print(result)

# # 模型启动web时使用
if __name__=='__main__':
    model.run()

#1. cpu推理单次耗时约2.7秒
#2. 模型占用内存648MiB
#3. 模型大小：82M