
import io,sys,os,base64,pysnooper
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import numpy
import os
import platform
from flask import g
from transformers import AutoTokenizer, AutoModel

class Chatglm_Model(Model):
    # 模型基础信息定义
    name='chatglm'   # 该名称与目录名必须一样，小写
    label='中文大模型chatglm'
    describe="中文大模型chatglm"
    field="自然语言"  # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes="聊天机器人"
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(Field_type.text, name='query', label='你的问题', describe='你的问题，最长200字',default='请问cube-studio是什么？',validators=Validator(max=200))
    ]

    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "lable": "示例一描述",
            "input": {
                "query": '请问cube-studio是什么？'
            }
        }
    ]

    # 加载模型，所有一次性的初始化工作可以放到该方法下。注意save_model_dir必须和训练函数导出的模型结构对应
    def load_model(self,save_model_dir=None,**kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("/models/", trust_remote_code=True)
        model = AutoModel.from_pretrained("/models/",trust_remote_code=True).half().quantize(4).cuda()
        self.model = model.eval()
        self.history={}

    # web每次用户请求推理，用于对接web界面请求
    # @pysnooper.snoop()
    def inference(self,query,**kwargs):
        history = self.history.get(g.username,[]) if g.username else []
        response, history = self.model.chat(self.tokenizer, query, history=history)
        print(f"回答：{response}")

        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back=[
            {
                "text":response
            }
        ]
        return back

model=Chatglm_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(arg1='测试输入文本',arg2='test.jpg')  # 测试
# print(result)

# 模型启动web时使用
if __name__=='__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py web --save_model_dir xx
    model.run()

