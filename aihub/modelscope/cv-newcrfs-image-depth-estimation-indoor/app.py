import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_NEWCRFS_IMAGE_DEPTH_ESTIMATION_INDOOR_Model(Model):
    # 模型基础信息定义
    name='cv-newcrfs-image-depth-estimation-indoor'   # 该名称与目录名必须一样，小写
    label='基于神经窗口全连接CRFs的单目深度估计'
    describe="单目深度估计是从单张RGB图预测场景深度，是一个很具有挑战性的任务。现在做这个任务的方法大都是设计越来越复杂的网络来简单粗暴地回归深度图，但我们采取了一个更具可解释性的路子，就是使用优化方法中的条件随机场（CRFs）。由于CRFs的计算量很大，通常只会用于计算相邻节点的能量，而很难用于计算整个图模型中所有节点之间的能量。为了借助这种全连接CRFs的强大表征力，我们采取了一种折中的方法，即将整个图模型划分为一个个小窗口，在每个窗口里面进行全连接CRFs的计算，这样就可以大大减少计算量，使全连接CRFs在深度估计这一任务上成为了可能。同时，为了更好地在节点之间进行信息传递，我们利用多头注意力机制计算了多头能量函数，然后用网络将这个能量函数优化到一个精确的深度图。基于此，我们用视觉transformer作为encoder，神经窗口全连接条件随机场作为decoder，构建了一个bottom-up-top-down的网络架构，这个网络在KITTI、NYUv2上都取得了SOTA的性能，同时可以应用于全景图深度估计任务，在MatterPort3D上也取得了SOTA的性能。"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "4882"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_newcrfs_image-depth-estimation_indoor/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='arg0', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_depth_estimation.jpg"
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
        
        self.p = pipeline('image-depth-estimation', 'damo/cv_newcrfs_image-depth-estimation_indoor')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)

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

model=CV_NEWCRFS_IMAGE_DEPTH_ESTIMATION_INDOOR_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
model.load_model(save_model_dir=None)
result = model.inference(arg0='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_depth_estimation.jpg')  # 测试
print(result)

# # 模型启动web时使用
# if __name__=='__main__':
#     model.run()