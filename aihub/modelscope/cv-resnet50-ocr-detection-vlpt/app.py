import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import numpy
import pysnooper
import os
import cv2
class CV_RESNET50_OCR_DETECTION_VLPT_Model(Model):
    # 模型基础信息定义
    name='cv-resnet50-ocr-detection-vlpt'   # 该名称与目录名必须一样，小写
    label='读光-文字检测-单词检测模型-英文-VLPT预训练'
    describe="给定一张图片，检测出图内文字并给出多边形包围框。检测模型使用DB，backbone初始化参数基于多模态交互预训练方法VLPT。"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "915"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_resnet50_ocr-detection-vlpt/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例0",
            "input": {
                "image": "test.jpg"
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
        
        self.p = pipeline('ocr-detection', 'damo/cv_resnet50_ocr-detection-vlpt')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):
        result = self.p(image)
        text = result.get("polygons")
        #处理图片大小
        def resize_image(image):
            height, width = image.shape[:2]
            max_size = 1280
            print(height)
            print(width)
            if max(height, width) > max_size:
               if height > width:
                  ratio = max_size / height
               else:
                  ratio = max_size / width
               image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
            return image
        #给图片中识别出的单词画框
        img = cv2.imread(image)
        for index in text :
            
            pts = numpy.array(index, numpy.int32)
            img = resize_image(img)
            canvas = cv2.polylines(img, [pts], True, (0, 0, 255), 1)
        #创建文件保存目录
        os.makedirs('result',exist_ok=True)
        save_path = os.path.join('result',os.path.basename(image))
        # 写到另外的图片文件中即可
        cv2.imwrite(save_path, canvas)
        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back=[
            {
                "image": save_path,
                "text": str(text),
            }
        ]
        return back

model=CV_RESNET50_OCR_DETECTION_VLPT_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
#model.load_model(save_model_dir=None)
#result = model.inference(image='test.jpg')  # 测试
#print(result)

# # 模型启动web时使用
if __name__=='__main__':
     model.run()
#模型大小98M,内存占用1.22G,识别图片响应在4秒左右,没有GPU
#运行环境为腾讯云服务器	标准型S6 - 2核 4G,操作系统TencentOS Server 3.1 (TK4)
#识别结果比较准确
