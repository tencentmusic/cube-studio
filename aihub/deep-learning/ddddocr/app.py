import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type

import pysnooper
import os
import ddddocr
from PIL import Image
import cv2

class DDDDOCR_Model(Model):
    # 模型基础信息定义
    name='ddddocr'
    label='验证码识别'
    describe="ai识别验证码文字和验证码目标"
    field="机器视觉"
    scenes="图像识别"
    status='online'
    version='v20221001'
    pic='example.jpg'

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待识别图片', describe='用于验证码识别的图片')
    ]
    web_examples=[
        {
            "label":"示例1",
            "input":{
                "img_file_path": "test.jpg"
            }
        }
    ]

    # 加载模型
    def load_model(self,model_dir=None,**kwargs):
        # 模型
        self.ocr = ddddocr.DdddOcr(beta=True)
        self.dec = ddddocr.DdddOcr(det=True)

    # 推理
    # @pysnooper.snoop()
    def inference(self,img_file_path):
        # 验证码图片
        with open(img_file_path, 'rb') as f:
            image = f.read()
            print(type(image))

        # 目标检测
        poses = self.dec.detection(image)
        print(poses)

        result_im = cv2.imread(img_file_path)

        # 遍历检测出的文字
        result_text=''
        for box in poses:
            x1, y1, x2, y2 = box
            cropped = result_im[y1:y2, x1:x2]
            cv2.imwrite("temp.jpg", cropped)
            txt = self.ocr.classification(open('temp.jpg', 'rb').read())
            result_text+=txt
            # 给每个文字画矩形框
            result_im = cv2.rectangle(result_im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        if os.path.exists('tem.jpg'):
            os.remove('temp.jpg')

        os.makedirs('result', exist_ok=True)
        save_path = os.path.join('result', os.path.basename(img_file_path))
        cv2.imwrite(save_path, result_im)
        back=[{
            "image":save_path,
            "text":result_text
        }]
        return back

model=DDDDOCR_Model()
# model.load_model()
# result = model.inference(img_file_path='test.jpg')  # 测试
# print(result)

# # 启动服务
if __name__=='__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py inference --arg1 xx --arg2 xx
    # python app.py web
    model.run()

