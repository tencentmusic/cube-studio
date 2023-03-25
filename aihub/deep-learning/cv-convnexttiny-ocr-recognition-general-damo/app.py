import base64
import io, sys, os
from cubestudio.aihub.model import Model, Validator, Field_type, Field

import pysnooper
import os


class CV_CONVNEXTTINY_OCR_RECOGNITION_GENERAL_DAMO_Model(Model):
    # 模型基础信息定义
    name = 'cv-convnexttiny-ocr-recognition-general-damo'  # 该名称与目录名必须一样，小写
    label = '读光-文字识别-行识别模型-中英-通用领域'
    describe = "给定一张图片，识别出图中所含文字并输出字符串。"
    field = "机器视觉"
    scenes = ""
    status = 'online'
    version = 'v20221001'
    pic = 'example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "37469"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-general_damo/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='image', label='', describe='', default='', validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples = [
        {
            "label": "示例1",
            "input": {
                "image": "ocr_recognition.jpg"
            }
        }
    ]

    # 训练的入口函数，将用户输入参数传递
    def train(self, save_model_dir, arg1, arg2, **kwargs):
        pass

    # 加载模型
    def load_model(self, save_model_dir=None, **kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.p = pipeline('ocr-recognition', 'damo/cv_convnextTiny_ocr-recognition-general_damo')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self, image, **kwargs):

        import time
        result = self.p(image)
        back = [
            {
                "image": '',
                "text": result['text'],
                "video": '',
                "audio": '',
                "markdown": ''
            }
        ]

        # 输出运行时间
        run_time_ms = 0.0
        # max_run_time_ms = 0.0
        # min_run_time_ms = 1000000.0
        # for i in range(20):
        #     time_start = time.clock()
        #     self.p(image)
        #     # 获取结束时间
        #     time_end = time.clock()
        #     # 计算运行时间
        #     run_time = (time_end - time_start)*1000
        #     if run_time > max_run_time_ms:
        #         max_run_time_ms = run_time
        #     if run_time < min_run_time_ms:
        #         min_run_time_ms = run_time
        #     run_time_ms += run_time
        # print("运行时间：", run_time_ms/20, "ms")
        # print("最大运行时间：", max_run_time_ms, "ms")
        # print("最小运行时间：", min_run_time_ms, "ms")

        return back


model = CV_CONVNEXTTINY_OCR_RECOGNITION_GENERAL_DAMO_Model()
model.load_model()

# import time
# time.sleep(20)
# 测试后将此部分注释
# result = model.inference(
#     image='ocr_recognition.jpg')  # 测试

# 测试后打开此部分
if __name__=='__main__':
    model.run()

# Date 2023/03/25
# Test by: mingshanyu
# 测试环境 cpu Intel i7-12700 测试example.jpg(58.6kb 1150*126) 使用cpu
# 推理循环20次，平均单次耗时 256.37ms, 最高单次耗时 389.95ms，最低单次耗时 183.24ms
# 模型大小74M
# 模型内存占用 307.3M