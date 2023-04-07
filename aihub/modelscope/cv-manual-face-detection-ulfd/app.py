import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import numpy,time,cv2,random
import pysnooper
import os

class CV_MANUAL_FACE_DETECTION_ULFD_Model(Model):
    # 模型基础信息定义
    name='cv-manual-face-detection-ulfd'   # 该名称与目录名必须一样，小写
    label='ULFD人脸检测模型-tiny'
    describe="1M轻量级人脸检测模型。给定一张图片，返回图片中人脸位置的坐标。ULFD为轻量级人脸检测算法, 基于SSD框架手工设计了backbone结构，是业界开源的第一个1M人脸检测模型。当输入320x240分辨率的图片且未使用onnxruntime加速时，在CPU上跑需要50-60ms，当使用onnxruntime加速后，在CPU上仅需要8-11ms。"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpeg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "4672"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_manual_face-detection_ulfd/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='arg0', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例0",
            "input": {
                "arg0": "ulfd_face_detection.jpg"
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
        
        self.p = pipeline('face-detection', 'damo/cv_manual_face-detection_ulfd')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)

        img = cv2.imread(arg0)  # 读取图片

        for index, face in enumerate(result['boxes']):
            boxs = face
            cv2.rectangle(img, (int(boxs[0]), int(boxs[1])), (int(boxs[2]), int(boxs[3])), (0, 0, 255), 2)

        savePath = 'result/result_' + str(int(1000 * time.time())) + '.jpg'
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        if os.path.exists(savePath):
            os.remove(savePath)
        cv2.imwrite(savePath, img)

        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back = [
            {
                "image": savePath
            }
        ]
        return back

model=CV_MANUAL_FACE_DETECTION_ULFD_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(arg0='ulfd_face_detection.jpg')  # 测试
# print(result)

# 模型启动web时使用
if __name__=='__main__':
    model.run()

# 模型大小 1M
# gpu v100 推理速度 0.1s