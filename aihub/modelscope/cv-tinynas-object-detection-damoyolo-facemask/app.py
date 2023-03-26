import base64
import io,sys,os
import numpy
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os,random,cv2
from PIL import Image,ImageFont
from PIL import ImageDraw

myfont = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 20)


# def resize_image(image):
#     height, width = image.shape[:2]
#     max_size = 1280
#     if max(height, width) > max_size:
#         if height > width:
#             ratio = max_size / height
#         else:
#             ratio = max_size / width
#         image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
#     return image


class CV_TINYNAS_OBJECT_DETECTION_DAMOYOLO_FACEMASK_Model(Model):
    # 模型基础信息定义
    name='cv-tinynas-object-detection-damoyolo-facemask'   # 该名称与目录名必须一样，小写
    label='实时口罩检测-通用'
    describe="本模型为高性能热门应用系列检测模型中的实时口罩检测模型，基于面向工业落地的高性能检测框架DAMOYOLO，其精度和速度超越当前经典的YOLO系列方法。用户使用的时候，仅需要输入一张图像，便可以获得图像中所有人脸的坐标信息，以及是否佩戴口罩。更多具体信息请参考Model card。"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "1536"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_facemask/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='arg0', label='待识别图片', describe='用于目标识别的原始图片'),
        # Field(type=Field_type.text, name='rtsp_url', label='视频流的地址', describe='rtsp视频流的地址')
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_tinynas_object-detection_damoyolo_facemask/assets/demo/test_00000954.jpg"
            }
        },
        {
            "label": "示例2",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_tinynas_object-detection_damoyolo_facemask/assets/demo/test_00003592.jpg"
            }
        },
        {
            "label": "示例3",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_tinynas_object-detection_damoyolo_facemask/assets/demo/test_00001095.jpg"
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
        self.p = pipeline('domain-specific-object-detection', 'damo/cv_tinynas_object-detection_damoyolo_facemask')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)
        img = Image.open(arg0)
        dr = ImageDraw.Draw(img)
        for idx in range(len(result['scores'])):
            class_name = result['labels'][idx]
            x1, y1, x2, y2 = result['boxes'][idx][0], result['boxes'][idx][1], result['boxes'][idx][2], result['boxes'][idx][3]
            # 画矩形框
            dr.rectangle((x1, y1, x2, y2), outline=(46, 254, 46), width=5)
            dr.text((x1, y1-25), class_name, font=myfont, fill='blue')
        save_path='result/result'+str(random.randint(5,5000))+'.jpg'
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        w, h = img.size
        max_size = 1280
        if max(h, w) > max_size:
            if h > w:
                ratio = max_size/h
                w = w * ratio
                h = h * ratio
            else:
                ratio = max_size/w
                w = w * ratio
                h = h * ratio
        img.resize((int(w), int(h)))
        img.save(save_path)
        back=[
            {
                "image": save_path,
                "text": str(result['scores']) + str(result['labels']),
            }
        ]
        return back




model=CV_TINYNAS_OBJECT_DETECTION_DAMOYOLO_FACEMASK_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(arg0='/mnt/workspace/.cache/modelscope/damo/cv_tinynas_object-detection_damoyolo_facemask/assets/demo/test_00000954.jpg')  # 测试
# print(result)

# # 模型启动web时使用
if __name__=='__main__':
    model.run()

# 模型大小：124.94MB
# 模型效果：近距离识别率较高
# 推理性能:首次调用3s内/后续100ms以内
# 模型占用内存/推理服务占用内存/gpu占用显存：7MB/2.5G/1.5GB
# 巧妙使用方法：第一次调用后，后面推理速度会加快