import base64
import io, sys, os
from cubestudio.aihub.model import Model, Validator, Field_type, Field
import random, numpy
import cv2
import pysnooper
import os
from modelscope.outputs import OutputKeys


class CV_UNET_IMAGE_MATTING_Model(Model):
    # 模型基础信息定义
    name = 'cv-unet-image-matting'  # 该名称与目录名必须一样，小写
    label = 'BSHM人像抠图'
    describe = "人像抠图对输入含有人像的图像进行处理，无需任何额外输入，实现端到端人像抠图，输出四通道人像抠图结果。"
    field = "机器视觉"  # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes = ""
    status = 'online'
    version = 'v20221001'
    pic = 'example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "52924"
    frameworks = "tensorflow"
    doc = "https://modelscope.cn/models/damo/cv_unet_image-matting/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='image', label='人像图片', describe='', default='', validators=None),
        Field(type=Field_type.image, name='background', label='背景图片', describe='', default='', validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples = [
        {
            "label": "示例0",
            "input": {
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-matting/1.png"
            }
        },
        {
            "label": "示例1",
            "input": {
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-matting/2.png"
            }
        },
        {
            "label": "示例2",
            "input": {
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-matting/3.png"
            }
        }
    ]

    # 训练的入口函数，此函数会自动对接pipeline，将用户在web界面填写的参数传递给该方法
    def train(self, save_model_dir, arg1, arg2, **kwargs):
        pass
        # 训练的逻辑
        # 将模型保存到save_model_dir 指定的目录下

    # 加载模型，所有一次性的初始化工作可以放到该方法下。注意save_model_dir必须和训练函数导出的模型结构对应
    def load_model(self, save_model_dir=None, **kwargs):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.p = pipeline('portrait-matting', 'damo/cv_unet_image-matting')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self, img: numpy.ndarray, **kwargs) -> numpy.ndarray:
        return img

    def add_alpha_channel(self, img):
        """ 为jpg图像添加alpha通道 """

        b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
        alpha_channel = numpy.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

        img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
        return img_new

    def merge_img(self, jpg_img, png_img, x1, y1, x2, y2):
        """ 将png透明图像与jpg图像叠加
            y1,y2,x1,x2为叠加位置坐标值
        """
        # 判断jpg图像是否已经为4通道
        if jpg_img.shape[2] == 3:
            jpg_img = self.add_alpha_channel(jpg_img)

        '''
        当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
        这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
        '''
        yy1 = 0
        yy2 = png_img.shape[0]
        xx1 = 0
        xx2 = png_img.shape[1]

        if x1 < 0:
            xx1 = -x1
            x1 = 0
        if y1 < 0:
            yy1 = - y1
            y1 = 0
        if x2 > jpg_img.shape[1]:
            xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
            x2 = jpg_img.shape[1]
        if y2 > jpg_img.shape[0]:
            yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
            y2 = jpg_img.shape[0]

        # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
        alpha_png = png_img[yy1:yy2, xx1:xx2, 3] / 255.0
        alpha_jpg = 1 - alpha_png

        # 开始叠加
        for c in range(0, 3):
            jpg_img[y1:y2, x1:x2, c] = (
                    (alpha_jpg * jpg_img[y1:y2, x1:x2, c]) + (alpha_png * png_img[yy1:yy2, xx1:xx2, c]))

        return jpg_img

    # web每次用户请求推理，用于对接web界面请求
    # @pysnooper.snoop(watch_explode=('result'))
    def inference(self, image, background=None, **kwargs):
        result = self.p(image)
        os.makedirs('result', exist_ok=True)
        save_path = 'result/result%s.png' % random.randint(1, 1000)
        cv2.imwrite(save_path, result[OutputKeys.OUTPUT_IMG])
        if background:

            # 正常读入图像，并保留其通道数不变，png图像为4通道，jpg图像为3通道
            img_jpg = cv2.imread(background, cv2.IMREAD_UNCHANGED)
            img_png = cv2.imread(save_path, cv2.IMREAD_UNCHANGED)

            print(img_png.shape)   # 高度，宽度，通道数
            print(img_jpg.shape)
            # 缩放到等宽高
            if img_jpg.shape[1] / img_jpg.shape[0] > img_png.shape[1] / img_png.shape[0]:
                print('背景图更宽')
                radio = img_jpg.shape[0] / img_png.shape[0]
                # img_jpg.resize(int(img_jpg.shape[1] / radio), int(img_jpg.shape[0] / radio))
                img_jpg = cv2.resize(img_jpg,(int(img_jpg.shape[1] / radio), int(img_jpg.shape[0] / radio)),interpolation=cv2.INTER_CUBIC)
                x0 = (img_jpg.shape[1] - img_png.shape[1]) // 2
                img_jpg = img_jpg[0:img_png.shape[0], x0:x0 + img_png.shape[1]]  # 裁剪坐标为[y0:y1, x0:x1]
            else:
                print('背景图更高')
                radio = img_jpg.shape[1] / img_png.shape[1]
                # img_jpg.resize(int(img_jpg.shape[1] / radio), int(img_jpg.shape[0] / radio))
                img_jpg = cv2.resize(img_jpg,(int(img_jpg.shape[1] / radio), int(img_jpg.shape[0] / radio)),interpolation=cv2.INTER_CUBIC)
                y0 = (img_jpg.shape[0] - img_png.shape[0]) // 2
                img_jpg = img_jpg[y0:y0 + img_png.shape[0],0:img_png.shape[1]]  # 裁剪坐标为[y0:y1, x0:x1]

            # 开始叠加
            x1 = 0
            y1 = 0
            x2 = x1 + img_png.shape[1]
            y2 = y1 + img_png.shape[0]

            print(img_png.shape)
            print(img_jpg.shape)
            cv2.imwrite('test.jpg', img_jpg)

            res_img = self.merge_img(img_jpg, img_png, x1, y1, x2, y2)
            cv2.imwrite(save_path, res_img)
        else:
            cv2.imwrite(save_path, result[OutputKeys.OUTPUT_IMG])

        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back = [
            {
                "image": save_path
            }
        ]
        return back


model = CV_UNET_IMAGE_MATTING_Model()

# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(image='https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-matting/1.png',background='background.jpg')  # 测试
# print(result)

# 模型启动web时使用
if __name__=='__main__':
    model.run()