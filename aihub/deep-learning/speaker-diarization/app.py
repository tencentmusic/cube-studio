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
    name='speaker-diarization'
    label='声纹识别'
    describe="获取音频文件中多人的人声位置和时长"
    field="听觉"
    scenes="语音检测"
    status='online'
    version='v20221001'
    pic='example.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址

    inference_inputs = [
        Field(type=Field_type.audio, name='audio_path', label='待检测的语音文件',describe='仅支持wav格式，mp3格式会被转化为wav格式后再识别'),
        Field(type=Field_type.text, name='speaker_num', label='说话人数量',default='-1',describe='提前不知道说话人数量填-1，知道填具体的数字，知道至少和至多说话人数量使用min~max方式填写')
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "audio_path": "test.wav",
                "speaker_num":"-1"

            }
        }
    ]

    # 加载模型
    # @pysnooper.snoop(depth=2)
    def load_model(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=os.getenv('HUGGINGFACE_TOKEN',None)
        )

    def mp32wav(self,mp3path,wavpath):
        from pydub import AudioSegment
        import wave

        # 读取mp3的波形数据
        sound = AudioSegment.from_file(mp3path, format='MP3')

        # 将读取的波形数据转化为wav
        f = wave.open(wavpath, 'wb')
        f.setnchannels(1)  # 频道数
        f.setsampwidth(2)  # 量化位数
        f.setframerate(16000)  # 取样频率
        f.setnframes(len(sound._data))  # 取样点数，波形数据的长度
        f.writeframes(sound._data)  # 写入波形数据
        f.close()

    # 推理
    @pysnooper.snoop()
    def inference(self,audio_path,speaker_num='-1'):
        if '.mp3' in audio_path:
            self.mp32wav(audio_path,audio_path.replace('.mp3','.wav'))

        save_path = os.path.join('result', os.path.basename(audio_path).split('.')[0] + ".rttm")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # apply the pipeline to an audio file
        if speaker_num=='-1':
            diarization = self.pipeline(audio_path)
        elif '~' in speaker_num:
            diarization = self.pipeline(audio_path,min_speakers=int(speaker_num.split('~')[0]),max_speakers=int(speaker_num.split('~')[1]))
        else:
            diarization = self.pipeline(audio_path,num_speakers=int(speaker_num))

        with open(save_path, "w") as rttm:
            diarization.write_rttm(rttm)
        results = open(save_path,mode='r').read()
        results = ["说话人 %s 起点 %s 时长%s"%(result.split(' ')[7],result.split(' ')[3],result.split(' ')[4])  for result in results.split('\n') if result.strip()]
        back=[
            {
                "html":'</br>'.join(results)
            }
        ]
        return back

model=Speaker_diarization_Model()
# model.load_model()
# result = model.inference(audio_path='test.wav',speaker_num='2')  # 测试
# print(result)

if __name__=='__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py inference --arg1 xx --arg2 xx
    # python app.py web
    model.run()


