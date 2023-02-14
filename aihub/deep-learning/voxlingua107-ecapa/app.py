import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type,Validator

import pysnooper
import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier

class Voxlingua107_ecapa_Model(Model):
    # 模型基础信息定义
    name='voxlingua107-ecapa'
    label='语种识别'
    describe="识别语音中的语种，包含107种不同的语种"
    field="听觉"
    scenes="语种识别"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub' # 'https://帮助文档的链接地址'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址

    inference_inputs = [
        Field(type=Field_type.audio, name='audio_path', label='待检测的语音文件',describe='支持107种不同的语种')
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "audio_path": "test.wav"
            }
        }
    ]

    # 加载模型
    # @pysnooper.snoop(depth=2)
    def load_model(self):
        self.language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp",use_auth_token=os.getenv('HUGGINGFACE_TOKEN',None))

    # 推理
    @pysnooper.snoop()
    def inference(self,audio_path):

        signal = self.language_id.load_audio(audio_path)
        prediction = self.language_id.classify_batch(signal)
        # print(prediction)
        probabilitys = prediction[1].exp().numpy().tolist()
        langes = prediction[3]
        print(probabilitys,langes)
        new_result = langes
        if len(probabilitys)==len(langes):
            new_result = []
            for index in range(len(probabilitys)):
                new_result.append('语言 %s，概率 %s %%'%(langes[index],str(round(probabilitys[index],2)*100)))

        back=[
            {
                "html":"<br>".join(new_result)
            }
        ]
        return back

model=Voxlingua107_ecapa_Model()
# model.load_model()
# result = model.inference(audio_path='test.wav')  # 测试
# print(result)

if __name__ =="__main__":
    server = Server(model=model)
    server.server(port=8080)


