import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import numpy,time,cv2,random
import pysnooper
import os

class CV_UNET_IMAGE_FACE_FUSION_DAMO_Model(Model):
    # 模型基础信息定义
    name='cv-unet-image-face-fusion-damo'   # 该名称与目录名必须一样，小写
    label='图像人脸融合'
    describe="给定一张模板图像和一张用户图像，图像人脸融合模型能够自动地将用户图中的人脸融合到模板人脸图像中，生成一张包含用户图人脸特征的新图像。"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "1911"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_unet-image-face-fusion_damo/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='template', label='模板图片',describe='模板图片，结果会包含该图片的整体信息',default='',validators=None),
        Field(type=Field_type.image, name='user', label='用户图片',describe='用户图片，结果会包含该图片的人脸特征',default='',validators=None)
    ]
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例0",
            "input": {
                "template": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facefusion_template.jpg",
                "user": "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facefusion_user.jpg"
            }
        }
    ]

    # 训练的入口函数，此函数会自动对接pipeline，将用户在web界面填写的参数传递给该方法
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        pass
        # 训练的逻辑
        # 将模型保存到save_model_dir 指定的目录下


    # 加载模型，所有一次性的初始化工作可以放到该方法下。注意save_model_dir必须和训练函数导出的模型结构对应
    def load_model(self,save_model_dir=None,**kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        self.p = pipeline('image-face-fusion', 'damo/cv_unet-image-face-fusion_damo')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,template,user,**kwargs):
        result = self.p({"template": template, "user": user})

        print(result)
        save_path = f'result/result{random.randint(1, 1000)}.jpg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        cv2.imwrite(save_path, result['output_img'])

        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back = [
            {
                "image": save_path
            }
        ]
        return back

model=CV_UNET_IMAGE_FACE_FUSION_DAMO_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(template='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facefusion_template.jpg',user='user.jpg')  # 测试
# print(result)

# 模型启动web时使用
if __name__=='__main__':
    model.run()

# 模型大小 1.4G
# 模型cpu运行速度 2s
# 注意：人脸的 角度最好是正的，不太大和太小