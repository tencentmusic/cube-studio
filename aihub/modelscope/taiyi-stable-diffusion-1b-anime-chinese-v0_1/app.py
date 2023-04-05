import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class TAIYI_STABLE_DIFFUSION_1B_ANIME_CHINESE_V0.1_Model(Model):
    # 模型基础信息定义
    name='taiyi-stable-diffusion-1b-anime-chinese-v0.1'   # 该名称与目录名必须一样，小写
    label='太乙-Stable-Diffusion-1B-动漫-中文-v0.1'
    describe="首个开源的中文Stable Diffusion动漫模型，基于100万筛选过的动漫中文图文对训练。"
    field="未知"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "4625"
    frameworks = ""
    doc = "https://modelscope.cn/models/Fengshenbang/Taiyi-Stable-Diffusion-1B-Anime-Chinese-v0.1/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.text, name='text', label='输入prompt',describe='输入prompt',default='',validators=Validator(max=75))
    ]

    inference_resource = {
        "resource_gpu": "4"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例0",
            "input": {
                "text": "1个女孩,绿色头发,毛衣,看向阅图者,上半身,帽子,户外,下雪,高领毛衣"
            }
        },
        {
            "label": "示例1",
            "input": {
                "text": "1个男生,帅气,微笑,看着阅图者,简单背景,白皙皮肤,上半身,衬衫,短发,单人"
            }
        },
        {
            "label": "示例2",
            "input": {
                "text": "户外,天空,云,蓝天,无人,多云的天空,风景,日出,草原"
            }
        },
        {
            "label": "示例3",
            "input": {
                "text": "室内,杯子,书,无人,窗,床,椅子,桌子,瓶子,窗帘,阳光,风景,盘子,木地板,书架,蜡烛,架子,书堆,绿植,梯子,地毯,小地毯"
            }
        },
        {
            "label": "示例4",
            "input": {
                "text": "户外,天空,水,树,无人,夜晚,建筑,风景,反射,灯笼,船舶,建筑学,灯笼,船,反射水,东亚建筑"
            }
        },
        {
            "label": "示例5",
            "input": {
                "text": "建筑,科幻,城市,城市风景,摩天大楼,赛博朋克,人群"
            }
        },
        {
            "label": "示例6",
            "input": {
                "text": "无人,动物,(猫:1.5),高清,棕眼"
            }
        },
        {
            "label": "示例7",
            "input": {
                "text": "无人,动物,(兔子:1.5),高清,棕眼"
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
        
        self.p = pipeline('text-to-image-synthesis', 'Fengshenbang/Taiyi-Stable-Diffusion-1B-Anime-Chinese-v0.1')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,text,**kwargs):
        result = self.p(text)

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

model=TAIYI_STABLE_DIFFUSION_1B_ANIME_CHINESE_V0.1_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(text='1个女孩,绿色头发,毛衣,看向阅图者,上半身,帽子,户外,下雪,高领毛衣')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()