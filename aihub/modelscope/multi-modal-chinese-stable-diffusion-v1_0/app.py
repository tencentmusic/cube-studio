import base64
import io, sys, os
from datetime import datetime

from cubestudio.aihub.model import Model, Validator, Field_type, Field
import torch
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import pysnooper
import os


class MULTI_MODAL_CHINESE_STABLE_DIFFUSION_Model(Model):
    # 模型基础信息定义
    name = 'multi-modal-chinese-stable-diffusion-v1.0'  # 该名称与目录名必须一样，小写
    label = '中文StableDiffusion-通用领域'
    describe = ""
    field = "强化学习"  # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes = ""
    status = 'online'
    version = 'v20221001'
    pic = 'example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "7703"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/multi-modal_chinese_stable_diffusion_v1.0/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='text', label='输入prompt', describe='想要绘画的内容，支持中文输入。',
              default='', validators=Validator(max=50)),
        Field(type=Field_type.text_select, name='steps', label='推理步数', default='30',
              choices=[str(x) for x in range(1, 70)]),
    ]

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "text": "黄公望 渔村国画"
            }
        }
    ]

    # 训练的入口函数，此函数会自动对接pipeline，将用户在web界面填写的参数传递给该方法
    def train(self, save_model_dir, arg1, arg2, **kwargs):
        pass
        # 训练的逻辑
        # 将模型保存到save_model_dir 指定的目录下

    # 加载模型，所有一次性的初始化工作可以放到该方法下。注意save_model_dir必须和训练函数导出的模型结构对应
    def load_model(self, save_model_dir=None, **kwargs):
        task = Tasks.text_to_image_synthesis
        model_id = 'damo/multi-modal_chinese_stable_diffusion_v1.0'
        # 基础调用
        self.pipe = pipeline(task=task, model=model_id)

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self, img, **kwargs):
        return img

    # web每次用户请求推理，用于对接web界面请求
    # @pysnooper.snoop(watch_explode=('result'))
    def inference(self, text, steps, **kwargs):
        os.makedirs('result', exist_ok=True)
        steps = int(steps)
        output = self.pipe({'text': text, 'num_inference_steps': steps})
        time_str = datetime.now().strftime('%Y%m%d%H%M%S')
        cv2.imwrite(f'result/result-{time_str}.png', output['output_imgs'][0])
        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back = [
            {
                "image": f'result/result-{time_str}.png',
            }
        ]
        return back


model = MULTI_MODAL_CHINESE_STABLE_DIFFUSION_Model()

# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(text='黄公望 渔村国画')  # 测试
# print(result)

# 模型启动web时使用
if __name__ == '__main__':
    model.run()

#  时间：2023.03.24
#  测试人： 蒋李雾龙
#  模型大小：共4.75GB
#  推理占用：A10 GPU下约10G左右显存
#  推理速度：A10 GPU下约5s左右
#  推理结果：图片
#  推理结果保存路径：result/result-xxxxxxxxxxxxxx.png
#  推荐效果：可以用来生成一些国画，油画，素描等等
#  不建议：不建议用来生成人物，生成的人物会比较奇怪。
#  警示：CPU无法运行，需要GPU
