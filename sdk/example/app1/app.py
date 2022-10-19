import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type

import pysnooper
import os

class APP1_Model(Model):
    # 模型基础信息定义
    name='app1'
    label='示例应用'
    description="ai示例应用，详细描述，都会显示应用描述上"
    field="机器视觉"
    scenes="图像识别"
    status='online'
    version='v20221001'
    doc='https://帮助文档的链接地址'
    pic='https://应用描述的缩略图/可以直接使用应用内的图片文件地址'
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='推理函数的输入参数', describe='用于文本识别的原始图片')
    ]

    # 加载模型
    def load_model(self):
        self.model = load("/xxx/xx/a.pth")

    # 推理
    @pysnooper.snoop()
    def inference(self,img_file_path):
            result_img='test_target.jpg'
            result_text='cat,dog'
            back=[{
                "image":result_img,
                "text":result_text
            }]
            return back

model=APP1_Model(init_shell=False)
model.load_model()
result = model.inference(img_file_path='test.png')  # 测试
print(result)

# # 启动服务
server = Server(model=model)
server.web_examples.append({
    "img_file_path":"test.png"
})
server.server(port=8080)

