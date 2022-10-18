import base64
import io,sys,os
root_dir = os.path.split(os.path.realpath(__file__))[0] + '/../../src/'
sys.path.append(root_dir)   # 将根目录添加到系统目录,才能正常引用common文件夹

from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.app import Server,Field,Field_type
import pysnooper
import os

class GFPGAN_Model(Model):
    # 模型基础信息定义
    name='gfpgan'
    label='图片上色'
    description="图片上色"
    field="机器视觉"
    scenes="图像合成"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/DeOldify'
    pic='https://picx.zhimg.com/v2-e96dd757c96464427560a9b5e5b07bc3_720w.jpg'
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待识别图片', describe='用于文本识别的原始图片')
    ]

    # 加载模型
    def load_model(self):
        learn = gen_inference_wide(root_folder=Path('./DeOldify'), weights_name='ColorizeVideo_gen')
        self.deoldfly_model = MasterFilter([ColorizerFilter(learn=learn)], render_factor=10)

    # 推理
    @pysnooper.snoop()
    def inference(self,img_file_path):
        import cv2
        import numpy as np
        from PIL import Image

        img = cv2.imread(img_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        filtered_image = self.deoldfly_model.filter(
            pil_img, pil_img, render_factor=35, post_process=True
        )

        result_img = np.asarray(filtered_image)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        save_path = img_file_path[0:img_file_path.rindex('.')] + "_target" + img_file_path[img_file_path.rindex('.'):]
        cv2.imwrite(save_path, result_img)

        back=[{
            "image":save_path
        }]
        return back


model=GFPGAN_Model(init_shell=False)
model.load_model()
result = model.inference(img_file_path='test.png')  # 测试
print(result)

# # 启动服务
server = Server(model=model)
server.server(port=8080)

