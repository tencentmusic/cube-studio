import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import datetime
import time
import cv2
import os

class CV_CSPNET_VIDEO_OBJECT_DETECTION_STREAMYOLO_Model(Model):
    # 模型基础信息定义
    name='cv-cspnet-video-object-detection-streamyolo'   # 该名称与目录名必须一样，小写
    label='StreamYOLO实时视频目标检测-自动驾驶领域'
    describe="实时视频目标检测模型"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "4907"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_cspnet_video-object-detection_streamyolo/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.video, name='video', label='',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "0"
    }

    web_examples=[
        {
            "label": "示例1",
            "input": {
                "video": "/mnt/workspace/.cache/modelscope/damo/cv_cspnet_video-object-detection_streamyolo/res/test_vod_00.mp4"
            }
        },
        {
            "label": "示例2",
            "input": {
                "video": "/mnt/workspace/.cache/modelscope/damo/cv_cspnet_video-object-detection_streamyolo/res/test_vod_01.mp4"
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
        
        self.p = pipeline('video-object-detection', 'damo/cv_cspnet_video-object-detection_streamyolo')
    
    # 图片resize代码
    def resize_image(self, image):
        height, width = image.shape[:2]
        max_size = 1280
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        return image
    
    # 可视化代码
    def vis_det_img(self, img, labels, scores, boxes):
        def get_color(idx):
            idx = (idx + 1) * 3
            color = ((10 * idx) % 255, (20 * idx) % 255, (30 * idx) % 255)
            return color
        unique_label = list(set(labels))
        for idx in range(len(scores)):
            x1, y1, x2, y2 = boxes[idx]
            score = str("%.2f"%scores[idx])
            label = str(labels[idx])
            color = get_color(unique_label.index(label))
            line_width = int(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
            text_size = 0.001 * (img.shape[0] + img.shape[1]) / 2 + 0.3
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_width)
            cv2.putText(img, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_PLAIN, text_size, color, line_width)
            cv2.putText(img, score, (int(x1), int(y2) + 10 + int(text_size*5)),
                        cv2.FONT_HERSHEY_PLAIN, text_size, color, line_width)
        # 为了方便展示，此处对图片进行了缩放
        img = self.resize_image(img)
        return img

    def generate_video(self, result, video_path):
        cap = cv2.VideoCapture(video_path)
        frame = cap.read()[1]
        frame = self.resize_image(frame)
        save_path = 'result/output_{}.mp4'.format(int(1000*time.time()))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_writter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
        boxes = result['boxes']
        labels = result['labels']
        scores = result['scores']
        timestamps = result['timestamps']
        
        for idx, timestamp in enumerate(timestamps):
            # 根据时间戳对视频进行抽帧
            time_buffer = datetime.datetime.strptime(timestamp, '%H:%M:%S.%f')
            timestamp = (time_buffer - datetime.datetime(1900, 1, 1)).total_seconds() * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp)
            frame = cap.read()[1]
            frame = self.vis_det_img(frame, labels[idx], scores[idx], boxes[idx])
            video_writter.write(frame)
        video_writter.release()
        return save_path

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,video,**kwargs):
        result = self.p(video)
        save_path = self.generate_video(result, video)
            
        back=[
            {
                "text": str(result),
                "video": save_path
            }
        ]
        return back

model=CV_CSPNET_VIDEO_OBJECT_DETECTION_STREAMYOLO_Model()

# 测试后将此部分注释
# model.load_model()
# result = model.inference(video='/mnt/workspace/.cache/modelscope/damo/cv_cspnet_video-object-detection_streamyolo/res/test_vod_00.mp4')  # 测试
# print(result)

# 测试后打开此部分
if __name__=='__main__':
    model.run()

'''
Date: 2023-03-26
Tested by: 秋水泡茶
模型大小：420M
模型效果：基于StreamYOLO的实时通用检测模型，支持8类交通目标检测，自动驾驶场景交通目标预测/检测，对于非自动驾驶前置摄象机场景会出现明显检测性能下降的情况。
推理性能：以1920*1200，fps=30，时长为6s的视频为例，推理时间为29.192s，视频导出时间为29.915s。
测试环境：CPU：Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz, GPU：Tesla V100 32G
占用GPU显存：程序在加载后不会自动加载模型，推理时模型加载后占用显存为1871M。
巧妙使用方法：
注意：本工程因为前端视频上传功能存在问题，导致测试后无法和web前端调通，因此只能通过命令行测试。
'''