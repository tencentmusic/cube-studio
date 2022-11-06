import base64
import io,sys,os

from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type
from cubestudio.util.py_image import img_base64

import pysnooper
from paddleocr import PaddleOCR, draw_ocr
from PIL import ImageGrab, Image
import numpy

import os

class Paddleocr_Model(Model):
    # 模型基础信息定义
    name='paddleocr'
    label='ocr识别'
    describe="paddleocr提供的ocr识别"
    field="机器视觉"
    scenes="图像识别"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/paddleocr'
    pic='https://blog.devzeng.com/images/ios-tesseract-ocr/how-ocr.png'

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待识别图片', describe='用于文本识别的原始图片')
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "img_file_path":"test.png"
            }
        }
    ]

    # 加载模型
    def load_model(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch")

    # 推理
    # @pysnooper.snoop()
    def inference(self,img_file_path):
        np_img = numpy.array(Image.open(img_file_path))

        text = ''
        result = self.ocr.ocr(np_img, cls=True)  # cls：测试是否需要旋转180°，影响性能，90°以及270°，无需开启。
        if result:
            result = result[0]
            for one in result:
                boxe = one[0]
                txt = one[1][0]
                print(boxe,txt)
                text += txt + '\r\n'

            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            print(boxes)
            print(txts)
            print(scores)
            im_show = draw_ocr(np_img, boxes, font_path='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')  #
            im_show = Image.fromarray(im_show)

            base64_str,byte_data = img_base64(im_show)
            os.makedirs('result',exist_ok=True)
            save_path = os.path.join('result',os.path.basename(img_file_path))
            with open(save_path, "wb") as imgFile:
                imgFile.write(byte_data)
            back=[{
                "image":save_path,
                "text":text
            }]
            return back

model=Paddleocr_Model()
# model.load_model()
# result = model.inference(img_file_path='test.png')  # 测试
# print(result)

# # 启动服务
server = Server(model=model)
server.server(port=8080)

