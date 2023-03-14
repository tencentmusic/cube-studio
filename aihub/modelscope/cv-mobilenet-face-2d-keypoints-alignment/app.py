import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server
import cv2
import pysnooper
import os
import random

class CV_MOBILENET_FACE_2D_KEYPOINTS_ALIGNMENT_Model(Model):
    # 模型基础信息定义
    name='cv-mobilenet-face-2d-keypoints-alignment'   # 该名称与目录名必须一样，小写
    label='106点人脸关键点-通用领域-2D'
    describe="人脸2d关键点对齐模型"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "6166"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_mobilenet_face-2d-keypoints_alignment/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.image, name='arg0', label='带人脸的输入图像',describe='带人脸的输入图像',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/1.jpg"
            }
        },
        {
            "label": "示例2",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/2.jpg"
            }
        },
        {
            "label": "示例3",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/3.jpg"
            }
        },
        {
            "label": "示例4",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/4.jpg"
            }
        },
        {
            "label": "示例5",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/5.jpg"
            }
        },
        {
            "label": "示例6",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/6.jpg"
            }
        },
        {
            "label": "示例7",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/7.jpg"
            }
        },
        {
            "label": "示例10",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/10.jpg"
            }
        },
        {
            "label": "示例11",
            "input": {
                "arg0": "/mnt/workspace/.cache/modelscope/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/11.jpg"
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
        
        self.p = pipeline('face-2d-keypoints', 'damo/cv_mobilenet_face-2d-keypoints_alignment')

    # 推理
    # @pysnooper.snoop(watch_explode=('result'))
    def inference(self,arg0,**kwargs):
        result = self.p(arg0)
        
        img = cv2.imread(arg0)  # 读取图片
        # 根据图片大小，判断画多大的圈
        for face in result['keypoints']:
            max_x=max_y=0
            min_x=min_y=float("inf")
            
            x = [keypoint.tolist()[0] for keypoint in face]
            y = [keypoint.tolist()[1] for keypoint in face]
            max_x,min_x,max_y,min_y = max(x),min(x),max(y),min(y)
            # print(max_x,max_y,min_x,min_y)
            radius = int(max(max_x-min_x,max_y-min_y)/50)
            if radius<1:
                radius=1
            # print(radius)
            for keypoint in face:
                keypoint = keypoint.tolist()
                cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), radius, (0, 0, 255), -1)
        save_path=f'result/result{random.randint(1,1000)}.jpg'
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        cv2.imwrite(save_path, img)
        back=[
            {
                "image":save_path,
                # "text":json.dumps(result,indent=4,ensure_ascii=False),
            }
        ]
        return back

model=CV_MOBILENET_FACE_2D_KEYPOINTS_ALIGNMENT_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(arg0='/mnt/workspace/.cache/modelscope/damo/cv_mobilenet_face-2d-keypoints_alignment/resources/2.jpg')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
