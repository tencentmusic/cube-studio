import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import numpy,cv2,time,random
import pysnooper
import os

class CV_UNET_VIDEO_DEINTERLACE_Model(Model):
    # 模型基础信息定义
    name='cv-unet-video-deinterlace'   # 该名称与目录名必须一样，小写
    label='视频去场纹'
    describe="视频去场纹，相比较于yadif、DIN等SOTA方法，对于大运动和低画质场景下的场纹有较好的场纹去除能力"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "128"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_unet_video-deinterlace/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.video, name='arg0', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_deinterlace_test.mp4"
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
        
        self.p = pipeline('video-deinterlace', 'damo/cv_unet_video-deinterlace')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)
        print(result)
        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back = [
            {
                "video": result['output_video']
            }
        ]
        return back

model=CV_UNET_VIDEO_DEINTERLACE_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(arg0='https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_deinterlace_test.mp4')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()

# 模型大小 70M,
# 模型推理速度  gpu v100上 1s视频，推理耗时30s
# 运行时显存占用10G