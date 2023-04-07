import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CHATYUAN_LARGE_Model(Model):
    # 模型基础信息定义
    name='chatyuan-large'   # 该名称与目录名必须一样，小写
    label='元语功能型对话大模型'
    describe="元语功能型对话大模型这个模型可以用于问答、结合上下文做对话、做各种生成任务，包括创意性写作，也能回答一些像法律、新冠等领域问题。它基于PromptCLUE-large结合数亿条功能对话多轮对话数据进一步训练得到。"
    field="自然语言"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpeg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "19923"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/ClueAI/ChatYuan-large/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='input', label='prompt文本',describe='prompt文本',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "text": "用户：帮我写个请假条，我因为新冠不舒服，需要请假10天，请领导批准\\n小元："
            }
        },
        {
            "label": "示例2",
            "input": {
                "text": "用户：新冠什么症状？\\n小元：新冠是指新型冠状病毒，其症状包括发热、干咳、乏力、嗅味觉减退、呼吸困难等。\\n用户：可以吃什么药？\\n小元：根据您提供的病史，目前没有明确的抗新冠病毒的药物，建议您在家进行自我隔离，避免与他人接触，多喝开水，清淡易消化饮食，避免熬夜和过度劳累，适当进行户外活动。\\n用户：用什么后遗症么？\\n小元："
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
        
        self.p = pipeline('text2text-generation', 'ClueAI/ChatYuan-large')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,input,**kwargs):
        result = self.p(input)

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

model=CHATYUAN_LARGE_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(text='用户：帮我写个请假条，我因为新冠不舒服，需要请假10天，请领导批准\n小元：')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()