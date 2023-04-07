
import io,sys,os,base64,pysnooper
from cubestudio.aihub.model import Model,Validator,Field_type,Field
import numpy

class APP1_Model(Model):
    # 模型基础信息定义
    name='app1'   # 该名称与目录名必须一样，小写
    label='示例应用中文名'
    describe="ai示例应用，详细描述，都会显示应用描述上，支持markdown"
    field="机器视觉"  # [机器视觉，听觉，自然语言，多模态，强化学习，图论]
    scenes="图像识别"
    status='online'
    version='v20221001'
    pic='example.jpg'  # 离线图片，作为模型的样式图，330*180尺寸比例
    # 和train函数的输入参数对应，并且会对接显示到pipeline的模板参数中
    train_inputs = [
        Field(Field_type.text, name='arg1', label='训练函数的输入参数arg1', describe='arg1的详细说明，用于在任务界面展示',default='这里是默认值',validators=Validator(regex='[a-z]*')),
        Field(Field_type.text_select, name='arg2', label='训练函数的输入参数arg2', describe='arg2的详细说明，用于在任务界面展示', default='这里是默认值',choices=['choice1','choice2','choice3'])
    ]
    # 和inference函数的输入参数对应，并且会对接显示到web界面上
    inference_inputs = [

        Field(type=Field_type.text, name='arg1', label='文本类推理输入参数',
              describe='接收到的参数值为字符串',default='这里是默认值',validators=Validator(regex='[a-z]*')),
        Field(type=Field_type.image, name='arg2', label='图片类推理输入',
              describe='需要用户上传图片,传递到推理函数中将是图片本地地址，Validator/max，控制可以上传多少张照片',validators=Validator(max=2,required=False)),
        Field(type=Field_type.video, name='arg3', label='视频类推理输入',
              describe='需要用户上传视频文件,传递到推理函数中将是视频本地地址'),
        Field(type=Field_type.audio, name='arg4', label='音频类推理输入',
              describe='需要用户上传音频文件,传递到推理函数中将是音频本地地址'),
        Field(type=Field_type.text_select, name='arg5', label='文本 选项类输入',
              describe='单选/多选 组件，传递到推理函数的值为选中的可选项的值，也就是choices中的值，Validator/max控制单选还是多选',choices=['choice1','choice2','choice3'],
              default='choice2',validators=Validator(max=1)),
        Field(type=Field_type.image_select, name='arg6', label='图片 选项类输入',
              describe='单选/多选 图片 组件，传递到推理函数的值为选中的可选项的值，也就是choices中的值，Validator/max控制单选还是多选', choices=['风格1.jpg', '风格2.jpg'],
              default='风格2.jpg',validators=Validator(max=1)),
        Field(type=Field_type.text_select, name='arg7', label='文本 选项类输入',
              describe='用于在界面展示,多选组件，max控制多选', choices=['choice1', 'choice2', 'choice3'],
              default=['choice1'],validators=Validator(max=3)),
        Field(type=Field_type.image_select, name='arg8', label='图片 选项类输入',
              describe='用于在界面展示,多选组件，max控制多选', choices=['风格1.jpg', '风格2.jpg'],
              default=['风格1.jpg','风格2.jpg'],validators=Validator(max=2)),
        Field(type=Field_type.capture, name='arg9', label='摄像头作为输入',
              describe='摄像头采样图片作为输入，max控制每秒采样帧数，例如10，表示每秒采样10帧', validators=Validator(max=10))
    ]

    # 会显示在web界面上，让用户作为示例输入
    web_examples=[
        {
            "lable": "示例一描述",
            "input": {
                "arg1": '测试输入文本',
                "arg2": 'test.jpg',
                "arg3": 'https://pengluan-76009.sz.gfp.tencent-cloud.com/cube-studio%20install.mp4'
            }
        }
    ]

    # 训练的入口函数，此函数会自动对接pipeline，将用户在web界面填写的参数传递给该方法
    def train(self,save_model_dir,arg1,arg2, **kwargs):
        print(arg1,arg2,kwargs)
        # 训练的逻辑
        # 将模型保存到save_model_dir 指定的目录下


    # 加载模型，所有一次性的初始化工作可以放到该方法下。注意save_model_dir必须和训练函数导出的模型结构对应
    def load_model(self,save_model_dir=None,**kwargs):
        # self.model = load("/xxx/xx/a.pth")
        pass

    # rtsp流的推理,输入为cv2 img,输出也为处理后的cv2 img
    def rtsp_inference(self,img:numpy.ndarray,**kwargs)->numpy.ndarray:
        return img

    # web每次用户请求推理，用于对接web界面请求
    # @pysnooper.snoop()
    def inference(self,arg1,arg2=None,arg3=None,arg4=None,arg5=None,arg6=None,arg7=None,**kwargs):
        # save_path = os.path.join('result', os.path.basename(arg1))
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 将结果保存到result目录下面，gitignore统一进行的忽略。并且在结果中注意添加随机数，避免多人访问时，结果混乱
        result_img='result/result.jpg'
        result_text='cat,dog'
        result_video='https://pengluan-76009.sz.gfp.tencent-cloud.com/cube-studio%20install.mp4'
        result_audio = 'result/test.wav'
        result_markdown=open('test.md',mode='r').read()

        # 推理的返回结果只支持image，text，video，audio，html，markdown几种类型
        back=[
            {
                "image":result_img,
                "text":result_text,
                "video":result_video,
                "audio":result_audio,
                "html":'<a href="/frontend/aihub/model_market/model_all">查看全部</a>',
                "markdown":result_markdown
            },
            {
                "image": result_img,
                "text": result_text,
                "video": result_video,
                "audio": result_audio,
                "markdown":result_markdown
            }
        ]
        return back

model=APP1_Model()


# 容器中调试训练时
# save_model_dir = "result"
# model.train(save_model_dir = save_model_dir,arg1=None,arg2=None)  # 测试

# 容器中运行调试推理时
# model.load_model(save_model_dir=None)
# result = model.inference(arg1='测试输入文本',arg2='test.jpg')  # 测试
# print(result)

# 模型启动web时使用
if __name__=='__main__':
    # python app.py train --arg1 xx --arg2 xx
    # python app.py web --save_model_dir xx
    model.run()

