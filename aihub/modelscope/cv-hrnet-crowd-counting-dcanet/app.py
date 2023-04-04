import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
from modelscope.outputs import OutputKeys
from modelscope.utils.cv.image_utils import numpy_to_cv2img
import cv2
import pysnooper
import os
import random, numpy

class CV_HRNET_CROWD_COUNTING_DCANET_Model(Model):
    # 模型基础信息定义
    name='cv-hrnet-crowd-counting-dcanet'   # 该名称与目录名必须一样，小写
    label='DCANet人群密度估计-多域'
    describe="采用单一模型就可以同时针对多个不同域的数据进行精确预测，是multidomain crowd counting中经典的方法"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "7294"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_hrnet_crowd-counting_dcanet/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='',default='',validators=None)
    ]
    
    def resize_image(image):
      height, width = image.shape[:2]
      max_size = 1280
      if max(height, width) > max_size:
          if height > width:
              ratio = max_size / height
          else:
              ratio = max_size / width
          image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
      return image

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "image": "/mnt/workspace/.cache/modelscope/damo/cv_hrnet_crowd-counting_dcanet/resources/crowd_counting.jpg"
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
        
        self.p = pipeline('crowd-counting', 'damo/cv_hrnet_crowd-counting_dcanet')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):
        result = self.p(image)
        scores = result[OutputKeys.SCORES]
        print('scores:', scores)
        vis_img = result[OutputKeys.OUTPUT_IMG]
        vis_img = numpy_to_cv2img(vis_img)
        save_path = f'result/result{random.randint(1, 1000)}.jpg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        cv2.imwrite(save_path, vis_img)
        back=[
            {
                "image": save_path,
                "text": str(scores),

            }
        ]
        return back

model=CV_HRNET_CROWD_COUNTING_DCANET_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(image='/mnt/workspace/.cache/modelscope/damo/cv_hrnet_crowd-counting_dcanet/resources/crowd_counting.jpg')  # 测试
# print(result)

# # 模型启动web时使用
if __name__=='__main__':
    model.run()

# 模型大小：55MB
# 模型效果：近距离识别率较高
# 推理性能: 200ms以内
# 模型占用内存/推理服务占用内存/gpu占用显存：10MB/2.5G/3.2GB
# 巧妙使用方法：第一次调用后，后面推理速度会加快