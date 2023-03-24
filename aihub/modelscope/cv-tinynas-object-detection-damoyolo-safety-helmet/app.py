import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import numpy
import pysnooper
from PIL import ImageGrab, Image
import cv2

class CV_TINYNAS_OBJECT_DETECTION_DAMOYOLO_SAFETY_HELMET_Model(Model):
    # 模型基础信息定义
    name='cv-tinynas-object-detection-damoyolo-safety-helmet'   # 该名称与目录名必须一样，小写
    label='实时安全帽检测-通用'
    describe="本模型为高性能热门应用系列检测模型中的实时安全帽（头盔）检测模型，基于面向工业落地的高性能检测框架DAMOYOLO，其精度和速度超越当前经典的YOLO系列方法。用户使用的时候，仅需要输入一张图像，便可以获得图像中所有人头的坐标信息，以及是否佩戴安全帽（头盔）。更多具体信息请参考Model card。"
    field="机器视觉"    # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    hot = "1036"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_safety-helmet/summary"

    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = []

    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [
        Field(type=Field_type.image, name='arg0', label='任意输入图像',describe='任意输入图像',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }
    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "test.jpg"
            }
        },
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
        
        self.p = pipeline('domain-specific-object-detection', 'damo/cv_tinynas_object-detection_damoyolo_safety-helmet')

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img
    # web每次用户请求推理，用于对接web界面请求
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)

        txts = result.get("labels")
        #处理图片大小
        def resize_image(image):
            height, width = image.shape[:2]
            max_size = 1280
            print(height)
            print(width)
            if max(height, width) > max_size:
               if height > width:
                  ratio = max_size / height
               else:
                  ratio = max_size / width
               image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
            return image
        #给图片画框
        def box_label(input_path, res):
            def get_color(idx):
                idx = (idx + 1) * 3
                color = ((10 * idx) % 255, (20 * idx) % 255, (30 * idx) % 255)
                return color
            img = cv2.imread(input_path)
            unique_label = list(set(res['labels']))
            for idx in range(len(res['scores'])):
                x1, y1, x2, y2 = res['boxes'][idx]
                score = str("%.2f"%res['scores'][idx])
                label = str(res['labels'][idx])
                color = get_color(unique_label.index(label))
                line_width = int(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
                text_size = 0.001 * (img.shape[0] + img.shape[1]) / 2 + 0.3
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_width)
                cv2.putText(img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_PLAIN, text_size, color, line_width)
                cv2.putText(img, score, (int(x1), int(y2) + 10 + int(text_size*5)),
                    cv2.FONT_HERSHEY_PLAIN, text_size, color, line_width)
            
            image = resize_image(img)
            return image
        #图片存储路径
        os.makedirs('result',exist_ok=True)
        save_path = os.path.join('result',os.path.basename(arg0))
        # 写到另外的图片文件中即可
        cv2.imwrite(save_path, box_label(arg0,result))
        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back=[
            {
                "image": save_path,
                "text": str(txts),
            }
        ]
        return back
model=CV_TINYNAS_OBJECT_DETECTION_DAMOYOLO_SAFETY_HELMET_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
#model.load_model(save_model_dir=None)
#result = model.inference(arg0='test.jpg')  # 测试
#print(result)

# # 模型启动web时使用
if __name__=='__main__':
     model.run()

#模型识别一张图片响应速度再3秒以内

