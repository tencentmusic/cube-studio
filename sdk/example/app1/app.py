import base64
import io,sys,os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server,Field,Field_type,Validator

import pysnooper
import os

class APP1_Model(Model):
    # 模型基础信息定义
    name='app1'
    label='示例应用中文名'
    description="ai示例应用，详细描述，都会显示应用描述上，支持markdown"
    field="机器视觉"
    scenes="图像识别"
    status='online'
    version='v20221001'
    doc='https://github.com/tencentmusic/cube-studio/tree/master/aihub' # 'https://帮助文档的链接地址'
    pic='https://user-images.githubusercontent.com/20157705/170216784-91ac86f7-d272-4940-a285-0c27d6f6cd96.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    # 运行基础环境脚本
    init_shell='init.sh'

    inference_inputs = [
        Field(type=Field_type.text, name='arg1', label='推理函数的输入参数arg1',
              describe='arg1的详细说明，用于在界面展示',default='这里是默认值',validators=Validator(regex='[a-z]*')),
        Field(type=Field_type.image, name='arg2', label='推理函数的输入参数arg2',
              describe='arg2的详细说明，用于在界面展示,传递到推理函数中将是图片本地地址，最多可上传2张',validators=Validator(max=2,required=False)),
        Field(type=Field_type.video, name='arg3', label='推理函数的输入参数arg3',
              describe='arg3的详细说明，用于在界面展示,传递到推理函数中将是视频本地地址'),
        Field(type=Field_type.audio, name='arg4', label='推理函数的输入参数arg4',
              describe='arg4的详细说明，用于在界面展示,传递到推理函数中将是音频本地地址'),
        Field(type=Field_type.text_select, name='arg5', label='推理函数的输入参数arg5',
              describe='arg5的详细说明，用于在界面展示,单选组件',choices=['choice1','choice2','choice3'],
              default='choice2',validators=Validator(max=1)),
        Field(type=Field_type.image_select, name='arg6', label='推理函数的输入参数arg6',
              describe='arg6的详细说明，用于在界面展示,单选组件', choices=['风格1.jpg', '风格2.jpg'],
              default='风格2.jpg',validators=Validator(max=1)),
        Field(type=Field_type.text_select, name='arg7', label='推理函数的输入参数arg7',
              describe='arg7的详细说明，用于在界面展示,多选组件，最多可选3个', choices=['choice1', 'choice2', 'choice3'],
              default=['choice1'],validators=Validator(max=3)),
        Field(type=Field_type.image_select, name='arg8', label='推理函数的输入参数arg8',
              describe='arg8的详细说明，用于在界面展示,多选组件，最多可选2张', choices=['风格1.jpg', '风格2.jpg'],
              default=['风格2.jpg','风格2.jpg'],validators=Validator(max=2))
    ]

    # 加载模型
    def load_model(self):
        # self.model = load("/xxx/xx/a.pth")
        pass

    # 推理
    # @pysnooper.snoop()
    def inference(self,arg1,arg2=None,arg3=None,arg4=None,arg5=None,arg6=None,arg7=None,**kwargs):
        # save_path = os.path.join('result', os.path.basename(arg1))
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_img='result.jpg'
        result_text='cat,dog'
        result_video='https://pengluan-76009.sz.gfp.tencent-cloud.com/cube-studio%20install.mp4'
        result_audio = 'test.wav'
        back=[
            {
                "image":result_img,
                "text":result_text,
                "video":result_video,
                "audio":result_audio
            },
            {
                "image": result_img,
                "text": result_text,
                "video": result_video,
                "audio": result_audio
            }
        ]
        return back

model=APP1_Model()
model.load_model()
# result = model.inference(arg1='测试输入文本',arg2='test.jpg')  # 测试
# print(result)

# # 启动服务
server = Server(model=model)
server.web_examples.append({
    "arg1":'测试输入文本',
    "arg2":'test.jpg',
    "arg3": 'https://pengluan-76009.sz.gfp.tencent-cloud.com/cube-studio%20install.mp4'
})
server.server(port=8080)

