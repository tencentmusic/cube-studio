import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import time
import cv2
import os

class CV_CSPNET_IMAGE_OBJECT_DETECTION_YOLOX_Model(Model):
    # 模型基础信息定义
    name='cv-cspnet-image-object-detection-yolox'   # 该名称与目录名必须一样，小写
    label='实时目标检测-通用领域'
    describe="基于yolox小模型的通用检测模型"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "6899"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_cspnet_image-object-detection_yolox/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='image', label='',describe='图片路径',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "image": "/mnt/workspace/.cache/modelscope/damo/cv_cspnet_image-object-detection_yolox/test_object_0.jpeg"
            }
        }
    ]

    # 训练的入口函数，将用户输入参数传递
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        pass


    # 加载模型
    def load_model(self,save_model_dir=None,**kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        self.p = pipeline('image-object-detection', 'damo/cv_cspnet_image-object-detection_yolox')

    def resize_image(self, image):
        height, width = image.shape[:2]
        max_size = 1280
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        return image

    # 可视化代码
    def vis_det_img(self, input_path, res):
        def get_color(idx):
            idx = (idx + 1) * 3
            color = ((10 * idx) % 255, (20 * idx) % 255, (30 * idx) % 255)
            return color
        img = cv2.imread(input_path)
        unique_label = list(set(res['labels']))
        for idx in range(len(res['scores'])):
            x1, y1, x2, y2 = res['boxes'][idx]
            score = str("%.2f"%res['scores'][idx])
            label = str(res['labels'][idx])
            color = get_color(unique_label.index(label))
            line_width = int(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
            text_size = 0.001 * (img.shape[0] + img.shape[1]) / 2 + 0.3
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_width)
            cv2.putText(img, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_PLAIN, text_size, color, line_width)
            cv2.putText(img, score, (int(x1), int(y2) + 10 + int(text_size*5)),
                        cv2.FONT_HERSHEY_PLAIN, text_size, color, line_width)
        # 为了方便展示，此处对图片进行了缩放
        img = self.resize_image(img)
        return img
    
    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,image,**kwargs):
        image_input = self.resize_image(cv2.imread(image))
        cv2.imwrite(image, image_input)
        result = self.p(image)
        savePath = 'result/result_' + str(int(1000*time.time())) + '.jpg'
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        if os.path.exists(savePath):
            os.remove(savePath)
        image_vis = self.vis_det_img(image, result)
        cv2.imwrite(savePath, image_vis)

        back=[
            {
                "image": savePath,
                "text": str(result)
            }
        ]
        return back

model=CV_CSPNET_IMAGE_OBJECT_DETECTION_YOLOX_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference(image='/mnt/workspace/.cache/modelscope/damo/cv_cspnet_image-object-detection_yolox/test_object_0.jpeg')  # 测试
# print(result)

# 测试后打开此部分
if __name__=='__main__':
    model.run()

'''
Date: 2023-03-22
Tested by: 秋水泡茶
模型大小：69M
模型效果：支持80类通用目标检测，对图片中的大目标具有良好的检测效果，对小目标检测较困难。
推理性能：以1330*1330的图片为例，推理速度平均为300ms。以720*1280的图片为例，推理速度最快234ms，最慢为486ms。
测试环境：CPU：Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz, GPU：Tesla V100 32G
占用GPU显存：启动时内存占用约为1467M，推理时占用约为1577M。
巧妙使用方法：
'''