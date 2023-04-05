import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class NLP_STRUCTBERT_FACT_CHECKING_CHINESE_BASE_Model(Model):
    # 模型基础信息定义
    name='nlp-structbert-fact-checking-chinese-base'   # 该名称与目录名必须一样，小写
    label='StructBERT事实准确性检测-中文-电商-base'
    describe="StructBERT事实准确性检测-中文-电商-base是在structbert-base-chinese预训练模型的基础上，使用业务数据训练出的自然语言推理模型，用于事实准确性检测，输入两个句子，判断两个句子描述的事实是否一致。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpeg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "1038"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/nlp_structbert_fact-checking_chinese-base/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='arg0', label='',describe='',default='',validators=Validator(max=300)),
        Field(type=Field_type.text, name='arg1', label='',describe='',default='',validators=Validator(max=300))
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "不易生锈",
                "arg1": "这款水槽是不锈钢材质的，非常的耐用，日常使用都是不会生锈的不会掉色的。"
            }
        },
        {
            "label": "示例2",
            "input": {
                "arg0": "适合老人用",
                "arg1": "70直径 加大款 适合体重小于200斤"
            }
        },
        {
            "label": "示例3",
            "input": {
                "arg0": "好通透呀",
                "arg1": "透明片以矫正视力为主，透明片有无色透明或淡蓝色透明。"
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
        
        self.p = pipeline('nli', 'damo/nlp_structbert_fact-checking_chinese-base')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,arg1,**kwargs):
        result = self.p({"arg0": arg0, "arg1": arg1})

        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
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

model=NLP_STRUCTBERT_FACT_CHECKING_CHINESE_BASE_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(arg0='不易生锈',arg1='这款水槽是不锈钢材质的，非常的耐用，日常使用都是不会生锈的不会掉色的。')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()