import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type,Validator

import pysnooper
import os
from pyannote.audio import Pipeline

class Speaker_diarization_Model(Model):
    # 模型基础信息定义
    name='speaker_diarization'
    label='语音片段识别'
    describe="获取音频文件中语音的人生说话的位置"
    field="听觉"
    scenes="语音检测"
    status='offline'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub' # 'https://帮助文档的链接地址'
    pic=''  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址

    inference_inputs = [
        Field(type=Field_type.video, name='audio_path', label='待检测的语音文件')
    ]

    # 加载模型
    def load_model(self):
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2022.07")

    # 推理
    @pysnooper.snoop()
    def inference(self,audio_path):
        # apply the pipeline to an audio file
        diarization = self.pipeline(audio_path)
        save_path = os.path.join('result', os.path.basename(audio_path).split('.')[0]+".rttm")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as rttm:
            diarization.write_rttm(rttm)

        back=[
            {
                "text":save_path
            }
        ]
        return back

model=Speaker_diarization_Model()
model.load_model()
result = model.inference(audio_path='test.wav')  # 测试
print(result)

# # 启动服务
server = Server(model=model)
server.web_examples.append({
    "audio_path":'test.wav',
})
server.server(port=8080)

