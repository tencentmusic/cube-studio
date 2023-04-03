import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os
import numpy

class MPLUG_IMAGE_TEXT_RETRIEVAL_FLICKR30K_LARGE_EN_Model(Model):
    # 模型基础信息定义
    name='mplug-image-text-retrieval-flickr30k-large-en'   # 该名称与目录名必须一样，小写
    label='mPLUG图文检索模型-英文-large'
    describe="达摩MPLUG英文图文检索large模型"
    field="多模态"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "128"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/mplug_image-text-retrieval_flickr30k_large_en/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='image', label='图片',describe='图片',default='',validators=None),
        Field(type=Field_type.text, name='text', label='文本',describe='文本',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "image": "test.jpg",
                "text": "A red panda stands on the grass."
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
        
        self.p = pipeline('image-text-retrieval', 'damo/mplug_image-text-retrieval_flickr30k_large_en')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,text,**kwargs):
        result = self.p({"image": image, "text": text})
        text = result.get("scores")
        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back=[
            {
                "text": str(text),
            }
        ]
        return back

model=MPLUG_IMAGE_TEXT_RETRIEVAL_FLICKR30K_LARGE_EN_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
#model.load_model(save_model_dir=None)
#result = model.inference(image='test.jpg',text='A red panda stands on the grass.')  # 测试
#print(result)

# # 模型启动web时使用
if __name__=='__main__':
     model.run()
#模型大小4.9G,内存占用10.68G,识别图片响应在4秒左右,没有GPU
#占用内存较高,需至少16G内存
