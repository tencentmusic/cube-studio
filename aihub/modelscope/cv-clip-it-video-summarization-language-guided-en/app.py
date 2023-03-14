import base64
import io,sys,os
from cubestudio.aihub.model import Model,Validator,Field_type,Field

import pysnooper
import os

class CV_CLIP_IT_VIDEO_SUMMARIZATION_LANGUAGE_GUIDED_EN_Model(Model):
    # 模型基础信息定义
    name='cv-clip-it-video-summarization-language-guided-en'   # 该名称与目录名必须一样，小写
    label='CLIP_It自然语言引导的视频摘要-Web视频领域-英文'
    describe="自然语言引导的视频摘要，用户根据自己的需求输入一段自然语言和一个长视频，算法根据用户输入自然语言的内容对输入视频进行自适应的视频摘要。"
    field="机器视觉"
    scenes=""
    status='online'
    version='v20221001'
    pic='result.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    hot = "2537"
    frameworks = "pytorch"
    doc = "https://modelscope.cn/models/damo/cv_clip-it_video-summarization_language-guided_en/summary"

    train_inputs = []

    inference_inputs = [
        Field(type=Field_type.video, name='video', label='mp4视频',describe='',default='',validators=None),
        Field(type=Field_type.text, name='text', label='视频摘要，空格分割',describe='',default='',validators=None)
    ]

    inference_resource = {
        "resource_gpu": "1"
    }

    web_examples=[
        {
            "label": "",
            "input": {
                "video": "/mnt/workspace/.cache/modelscope/damo/cv_clip-it_video-summarization_language-guided_en/video/video_category_test_video.mp4",
                "text": "phone box"
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
        
        self.p = pipeline('language-guided-video-summarization', 'damo/cv_clip-it_video-summarization_language-guided_en')

    # 推理
    @pysnooper.snoop(watch_explode=('result'))
    def inference(self,video,text,**kwargs):
        text = re.split(',|;|\n|\t| |，|；', str(text))
        result = self.p(video,text)
        back=[
            {
                "image": 'result/aa.jpg',
                "text": '结果文本',
                "video": 'result/aa.mp4',
                "audio": 'result/aa.mp3',
                "markdown":''
            }
        ]
        return back

model=CV_CLIP_IT_VIDEO_SUMMARIZATION_LANGUAGE_GUIDED_EN_Model()

# 测试后将此部分注释
model.load_model()
result = model.inference(video='/mnt/workspace/.cache/modelscope/damo/cv_clip-it_video-summarization_language-guided_en/video/video_category_test_video.mp4',text='phone box')  # 测试
print(result)

# 测试后打开此部分
# if __name__=='__main__':
#     model.run()
