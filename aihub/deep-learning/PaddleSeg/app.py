import io, sys, os
from cubestudio.aihub.model import Model
from cubestudio.aihub.web.server import Server, Field, Field_type
import pysnooper
from datetime import datetime

try:
    os.system("ln -f -s /Matting /app/")
except:
    print("软连接出错，请重试！")

from Matting.tools.bg_replace import parse_args, prepare, run_job
import traceback


class PaddleMatting_Model(Model):
    # 模型基础信息定义
    name = 'PaddleMatting'
    label = '人像抠图'
    describe = "证件照制作，证件照背景替换，可将照片修改成证件照。人像抠图后，可自定义人像背景。"
    field = "机器视觉"
    scenes = "人像抠图"
    status = 'online'
    version = 'v20230317'
    pic = 'https://user-images.githubusercontent.com/30919197/179751613-d26f2261-7bcf-4066-a0a4-4c818e7065f0.gif'


    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='需要修改的图片', describe='用于更换背景的原始图片'),
        Field(type=Field_type.text_select, name='style', label='图片背景颜色', default='白色',
              choices=['白色', '蓝色', '绿色', '红色',], describe='背景颜色'),
        Field(type=Field_type.image, name='bg_img_file_path', label='背景图片', describe='更换背景为此图片效果（此项存在时，背景纯色修改将失效！）'),
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "img_file_path": "test.png"
            }
        }
    ]


    # 加载模型
    def load_model(self, **kwargs):
        args = parse_args()
        self.model, self.cfg = prepare(args)


    # 推理
    # @pysnooper.snoop()
    def inference(self, img_file_path, style, bg_img_file_path):
        styles = {'白色': 'w', '蓝色': 'b', '绿色': 'g', '红色': 'r', }
        try:
            if bg_img_file_path:
                style = bg_img_file_path
            else:
                style = styles[style]
            os.makedirs('result/', exist_ok=True)
            save_path = run_job(self.model, self.cfg, img_file_path, style)
            back = [{
                "image": save_path,
            }]
            return back
        except Exception as ex:
            print(ex)
            print(traceback.format_exc())


model = PaddleMatting_Model()
# model.load_model()
# result = model.inference(img_file_path='test.jpg')  # 测试
# print(result)

if __name__ == '__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py inference --arg1 xx --arg2 xx
    # python app.py web
    model.run()
