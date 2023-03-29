import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os
import numpy

class SPRING_COUPLET_GENERATION_Model(Model):
    # 模型基础信息定义
    name='spring-couplet-generation'   # 该名称与目录名必须一样，小写
    label='春联生成模型-中文-base'
    describe="春联生成模型是达摩院AliceMind团队利用基础生成大模型在春联场景的应用，该模型可以通过输入两字随机祝福词，生成和祝福词相关的春联。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "12928"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/spring_couplet_generation/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='text', label='以关键词“xxxx”写一副春联',describe='以关键词“xxxx”写一副春联',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "text": "以关键词“幸福”写一副春联"
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
        
        self.p = pipeline('text-generation', 'damo/spring_couplet_generation')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,text,**kwargs):
        result = self.p(text)
        text = result.get("text")
        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back=[
            {
                "text": str(text),
            }
        ]
        return back

model=SPRING_COUPLET_GENERATION_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
#model.load_model(save_model_dir=None)
#result = model.inference(text='以关键词“幸福”写一副春联')  # 测试
#print(result)

# # 模型启动web时使用
if __name__=='__main__':
     model.run()
#模型大小为1.2G,运行内存占用为3.14G,响应速度在5秒以内,没有GPU
#运行环境为腾讯云服务器	标准型SA3 - 4核 8G,操作系统TencentOS Server 3.1 (TK4)
#须用<以关系词"xx"写一副春联>的格式输入,不然生成的春联有问题
