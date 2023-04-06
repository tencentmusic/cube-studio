import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os
import numpy

class MPLUG_VISUAL_QUESTION_ANSWERING_COCO_BASE_ZH_Model(Model):
    # 模型基础信息定义
    name='mplug-visual-question-answering-coco-base-zh'   # 该名称与目录名必须一样，小写
    label='mPLUG视觉问答模型-中文-base'
    describe="本任务是mPLUG在中文VQA进行finetune的视觉问答下游任务，给定一个问题和图片，通过图片信息来给出答案。"
    field="多模态"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "4068"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/mplug_visual-question-answering_coco_base_zh/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='image', label='图片',describe='图片',default='',validators=None),
        Field(type=Field_type.text, name='question', label='问题',describe='问题',default='',validators=None)
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
                "question": "这是什么动物？"
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
        
        self.p = pipeline('visual-question-answering', 'damo/mplug_visual-question-answering_coco_base_zh')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,question,**kwargs):
        result = self.p({"image": image, "question": question})
        text = result.get("text")
        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back=[
            {
                "text": str(text),
            }
        ]
        return back

model=MPLUG_VISUAL_QUESTION_ANSWERING_COCO_BASE_ZH_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
#model.load_model(save_model_dir=None)
#result = model.inference(image='test.jpg',question='这是什么动物？')  # 测试
#print(result)

# # 模型启动web时使用
if __name__=='__main__':
     model.run()
#模型大小1.8G,内存占用6.017G,识别图片响应在3秒内,没有GPU
#识别常见动物以及植物比较准确
