import base64
import io, sys, os
from cubestudio.aihub.model import Model, Validator, Field_type, Field
import time
import pysnooper
import os
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class CV_GAN_FACE_IMAGE_GENERATION_Model(Model):
    # 模型基础信息定义
    name = 'cv-gan-face-image-generation'  # 该名称与目录名必须一样，小写
    label = 'StyleGAN2人脸生成'
    describe = "StyleGAN是图像生成领域的代表性工作，StyleGAN2在StyleGAN的基础上，采用Weight Demodulation取代AdaIN等改进极大的减少了water droplet artifacts等，生成结果有了质的提升，甚至能达到以假乱真的程度。"
    field = "机器视觉"
    scenes = ""
    status = 'online'
    version = 'v20221001'
    pic = 'example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "10633"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_gan_face-image-generation/summary"

    train_inputs = []

    inference_resource = {
        "resource_gpu": "1"
    }

    # 训练的入口函数，将用户输入参数传递
    def train(self, save_model_dir, arg1, arg2, **kwargs):
        pass

    # 加载模型
    def load_model(self, save_model_dir=None, **kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.p = pipeline('face-image-generation', 'damo/cv_gan_face-image-generation')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self, **kwargs):
        seed = int(time.time()*1000)
        result = self.p(seed)
        save_path = 'result/result-%s.jpg'%seed
        os.makedirs('result',exist_ok=True)
        cv2.imwrite(save_path, result[OutputKeys.OUTPUT_IMG])
        back = [
            {
                "image": save_path
            }
        ]
        return back


model = CV_GAN_FACE_IMAGE_GENERATION_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference()  # 测试
# print(result)

# 测试后打开此部分
if __name__=='__main__':
    model.run()


# 模型大小128M
# cpu模型推理时长  1.5s