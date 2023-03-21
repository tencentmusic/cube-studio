import io, sys, os


from cubestudio.aihub.model import Model
from cubestudio.aihub.web.server import Server, Field, Field_type
import pysnooper
from datetime import datetime

try:
    os.system("ln -f -s /PaddleDetection/* /app/")
except:
    print("软连接出错，请重试！")

from tools.infer import main, pre_trainer, do_work
import traceback


class PaddleDetection_Model(Model):
    # 模型基础信息定义
    name = 'paddle-detection'
    label = '目标识别'
    describe = "Paddle中YOLO部分 目标检测"
    field = "机器视觉"
    scenes = "目标识别"
    status = 'online'
    version = 'v20230314'
    pic = 'example.jpg'


    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待识别图片', describe='用于目标识别的原始图片'),
        # Field(type=Field_type.text, name='rtsp_url', label='视频流的地址', describe='rtsp视频流的地址')
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "img_file_path": "test.jpg"
            }
        }
    ]


    # 加载模型
    def load_model(self, **kwargs):
        flags, cfg = main()
        self.trainer, self.flags = pre_trainer(flags, cfg)

    # rtsp流的推理
    # @pysnooper.snoop()
    def rtsp_inference(self, img, **kwargs):
        # 后续支持
        pass


    # 推理
    # @pysnooper.snoop()
    def inference(self, img_file_path, rtsp_url=None):
        try:
            os.makedirs('result/', exist_ok=True)
            results, save_name = do_work(self.trainer, self.flags, img_file_path)
            back = [{
                "text": results,
                "image": save_name,
            }]
            return back
        except Exception as ex:
            print(ex)
            print(traceback.format_exc())


model = PaddleDetection_Model()
# model.load_model()
# result = model.inference(img_file_path='test.jpg')  # 测试
# print(result)

if __name__ == '__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py inference --arg1 xx --arg2 xx
    # python app.py web
    model.run()
