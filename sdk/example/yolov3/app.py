import base64
import io,sys,os
root_dir = os.path.split(os.path.realpath(__file__))[0] + '/../../src/'
sys.path.append(root_dir)   # 将根目录添加到系统目录,才能正常引用common文件夹

from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type

import pysnooper
from darknetpy.detector import Detector
from PIL import ImageGrab, Image
from PIL import Image,ImageFont
from PIL import ImageDraw
import numpy

import os
myfont = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 20)

class Yolov3_Model(Model):
    # 模型基础信息定义
    name='yolov3'
    label='目标识别'
    description="darknet yolov3 目标识别"
    field="机器视觉"
    scenes="目标识别"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/paddleocr'
    pic='https://blog.devzeng.com/images/ios-tesseract-ocr/how-ocr.png'
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待识别图片', describe='用于文本识别的原始图片')
    ]

    # 加载模型
    def load_model(self):
        YOLO_DATA_PATH = os.getenv('YOLO_DATA_PATH', 'yolo/coco.data')
        YOLO_CFG_PATH = os.getenv('YOLO_CFG_PATH', 'yolo/yolov3.cfg')
        YOLO_WEIGHTS_PATH = os.getenv('YOLO_WEIGHTS_PATH', 'yolo/yolov3.weights')
        self.detector = Detector(YOLO_DATA_PATH, YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)

    # 推理
    @pysnooper.snoop()
    def inference(self,img_file_path):
        res = self.detector.detect(img_file_path)

        img = Image.open(img_file_path)
        dr = ImageDraw.Draw(img)
        for data in res:
            class_name = data['class']
            x, y, w, h = data['left'], data['top'], data['right'] - data['left'], data['bottom'] - data['top']
            # 画矩形框
            dr.rectangle((x, y, x + w, y + h), outline=(46, 254, 46), width=3)
            dr.text((data['left'], data['top']), class_name, font=myfont, fill='red')
        out_image_path = img_file_path[:img_file_path.rfind('.')] + '_target' + img_file_path[img_file_path.rfind('.'):]
        img.save(out_image_path)

        back=[{
            "image":out_image_path
        }]
        return back

model=Yolov3_Model(init_shell=False)
model.load_model()
result = model.inference(img_file_path='test.png')  # 测试
print(result)

# # # 启动服务
# server = Server(model=model)
# server.server(port=8080)

