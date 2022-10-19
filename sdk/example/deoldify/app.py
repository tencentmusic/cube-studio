import base64
import io,sys,os
import shutil

root_dir = os.path.split(os.path.realpath(__file__))[0] + '/../../src/'
sys.path.append(root_dir)   # 将根目录添加到系统目录,才能正常引用common文件夹

from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type


import pysnooper
from deoldify.visualize import *

import os

class DeOldify_Model(Model):
    # 模型基础信息定义
    name='deoldify'
    label='图片上色'
    description="图片上色"
    field="机器视觉"
    scenes="图像合成"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/DeOldify'
    pic='https://picx.zhimg.com/v2-e96dd757c96464427560a9b5e5b07bc3_720w.jpg?source=172ae18b'
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待识别图片', describe='用于文本识别的原始图片')
    ]

    # 加载模型
    def load_model(self):
        plt.style.use('dark_background')
        torch.backends.cudnn.benchmark = True
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
        self.colorizer = get_image_colorizer(root_folder=Path('/DeOldify'),artistic=True)

    # 推理
    @pysnooper.snoop()
    def inference(self,img_file_path):
        render_factor = 35
        save_path = self.colorizer.plot_transformed_image(path=img_file_path, render_factor=render_factor, compare=True)
        back=[{
            "image":str(save_path)
        }]
        return back


model=DeOldify_Model(init_shell=False)
model.load_model()
result = model.inference(img_file_path='test.png')  # 测试
print(result)

# 启动服务
server = Server(model=model)
server.web_examples.append(
    {"img_file_path":"test.png"}
)
server.server(port=8080)

