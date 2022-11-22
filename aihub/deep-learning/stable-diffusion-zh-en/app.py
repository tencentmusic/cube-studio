import io, sys, os
import random
from datetime import datetime
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server, Field, Field_type
import torch
import pysnooper
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor


class SD_ZH_Model(Model):
    # 模型基础信息定义
    name = 'stable-diffusion-zh'
    label = '文字转图像-中英文混合9种语言'
    describe = "输入一串文字描述，可生成相应的图片，暂已支持语言：英语(En)、中文(Zh)、西班牙语(Es)、法语(Fr)、俄语(Ru)、日语(Ja)、韩语(Ko)、阿拉伯语(Ar)和意大利语(It)"
    field = "神经网络"
    scenes = "图像创作"
    status = 'online'
    version = 'v20221122'
    doc = 'https://github.com/CompVis/stable-diffusion'  # 'https://帮助文档的链接地址'
    pic = 'https://images.nightcafe.studio//assets/stable-tile.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址

    inference_resource = {
        "resource_gpu": "1"
    }

    inference_inputs = [
        Field(type=Field_type.text, name='prompt', label='输入的文字内容',
              describe='输入的文字内容，支持9种语言输入，描述越详细越好', default='a photograph of an astronaut riding a horse'),
        # Field(type=Field_type.text_select, name='ddim_steps', label='推理的次数',
        #       describe='推理进行的次数，推荐20-50次将会得到更接近真实的图片', default="50",choices=["20","30","40","50","60","70"]),
        Field(type=Field_type.text_select, name='n_samples', label='推理出的图像数量',
              describe='结果中所展示的图片数量，数量越多则会导致性能下降', default="1", choices=[str(x + 1) for x in range(20)])
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "prompt": 'a photograph of an astronaut riding a horse',
                "ddim_steps": 50,
                "n_samples": 1
            }
        }
    ]

    # 加载模型
    # @pysnooper.snoop()
    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = AutoLoader(task_name="text2img",  # contrastive learning
                            model_name="AltDiffusion-m9",
                            model_dir="modeldir")

        model = loader.get_model()
        model.eval()
        model.to(device)
        self.predictor = Predictor(model)

    # 推理
    @pysnooper.snoop()
    def inference(self, prompt, n_samples=1, fixed_code=True, n_rows=0, **kwargs):
        try:
            time_str = datetime.now().strftime('%Y%m%d%H%M%S')
            if n_samples:
                n_samples = int(n_samples)
            seed = random.randint(0, 10000000)
            re_list = self.predictor.predict_generate_images(prompt, ddim_steps=40, n_samples=n_samples,
                                                             outpath='result', seed=seed, pic_name=time_str)
            back = [
                {
                    "image": img_path
                } for img_path in re_list
            ]
            return back
        except Exception as ex:
            print(ex)
            back = [{
                "text": f'出现错误，请联系开发人处理{str(ex)}'
            }]
            return back


model = SD_ZH_Model()
# model.load_model()
# result = model.inference(prompt='a photograph of an astronaut riding a horse')  # 测试
# print(result)

# 启动服务
server = Server(model=model)
server.server(port=8080)
