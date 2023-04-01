import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import time,random,cv2,numpy
import pysnooper
import os

class CV_ADAINT_IMAGE_COLOR_ENHANCE_MODELS_Model(Model):
    # 模型基础信息定义
    name='cv-adaint-image-color-enhance-models'   # 该名称与目录名必须一样，小写
    label='Adaptive-Interval-3DLUT图像调色'
    describe=""
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "70"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_adaint_image-color-enhance-models/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='arg0', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_adaint_image-color-enhance-models/data/1.png"
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
        
        self.p = pipeline('image-color-enhancement', 'damo/cv_adaint_image-color-enhance-models')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)

        from modelscope.outputs import OutputKeys
        savePath = 'result/result_' + str(int(1000 * time.time())) + '.jpg'
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        if os.path.exists(savePath):
            os.remove(savePath)
        cv2.imwrite(savePath, result[OutputKeys.OUTPUT_IMG])
        back = [
            {
                "image": savePath
            }
        ]
        return back

model=CV_ADAINT_IMAGE_COLOR_ENHANCE_MODELS_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference(arg0='/mnt/workspace/.cache/modelscope/damo/cv_adaint_image-color-enhance-models/data/1.png')  # 测试
# print(result)

# 测试后打开此部分
if __name__=='__main__':
    model.run()

# 模型大小 10M
# 模型 cpu 运行效率  0.1s
