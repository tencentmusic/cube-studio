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
    description="ai识别验证码文字和验证码目标"
    field="机器视觉"
    scenes="图像识别"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/ddddocr'
    pic='https://user-images.githubusercontent.com/20157705/191401572-43eb066c-e1cb-451b-8656-260df3a7b0e3.png'
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待识别图片', describe='用于验证码识别的图片')
    ]

    # 加载模型
    def load_model(self):
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

model=DDDDOCR_Model(init_shell=False)
model.load_model()
result = model.inference(img_file_path='test.png')  # 测试
print(result)

# # 启动服务
server = Server(model=model)
server.web_examples.append({
    "img_file_path":"test.png"
})
server.server(port=8080)

