import io, sys, os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server, Field, Field_type
import pysnooper
from datetime import datetime
import torch
import torch.distributed as dist
from yolov5.train import run


class Yolov5_Model(Model):
    # 模型基础信息定义
    name = 'yolov5'
    label = '目标识别'
    describe = "darknet yolov5 目标识别"
    field = "机器视觉"
    scenes = "目标识别"
    status = 'online'
    version = 'v20221118'
    doc = 'https://github.com/tencentmusic/cube-studio/tree/master/aihub/deep-learning/yolov5'
    pic = 'example.jpg'

    train_inputs = [
        Field(Field_type.str, name='data', label='数据地址', describe='数据地址'),
        Field(Field_type.str, name='weights', label='模型存储地址', describe='权重存储地址'),
        Field(Field_type.str, name='cfg', label='配置文件地址', describe='配置文件地址'),
        Field(Field_type.int, name='epochs', label='共进行的循环次数', describe='循环次数'),
        Field(Field_type.int, name='workers', label='加载数据工作数量', describe='加载数据工作数量'),
    ]
    # 初始 已包含 可以直接使用 (注：范围值仅供参考，并非上限)
    # data='yolov5/data/voc_ball.yaml'
    # weights='yolov5/yolov5s.pt'
    # cfg='yolov5/models/yolov5s_ball.yaml'
    # epochs=50-300
    # workers=1-8

    inference_inputs = [
        Field(type=Field_type.image, name='img_file_path', label='待识别图片', describe='用于目标识别的原始图片')
    ]
    web_examples = [
        {
            "label": "示例1",
            "input": {
                "img_file_path": "test.jpg"
            }
        }
    ]

    # 训练的入口函数，将用户输入参数传递
    def train(self, **kwargs):
        dist.init_process_group(backend='gloo')
        run(data=kwargs['data'], weights=kwargs['weights'], workers=kwargs['workers'], cfg=kwargs['cfg'],
            epochs=kwargs['epochs'])
        dist.destroy_process_group()

    # 加载模型
    def load_model(self):
        self.yolo_model = torch.hub.load('yolov5', 'yolov5s6', source='local', pretrained=True)

    # 推理
    @pysnooper.snoop()
    def inference(self, img_file_path):
        yolo_model = self.yolo_model
        os.makedirs('result', exist_ok=True)
        time_str = datetime.now().strftime('%Y%m%d%H%M%S')
        result = yolo_model(img_file_path)
        # result_text = result.print()
        # result.save(save_dir=f'result/{time_str}-{img_file_path.split(".")[0]}')
        # result_pic_dir = f"result/{time_str}-{img_file_path.split('.')[0]}/{img_file_path}"
        result_text = result.print()[:-2]
        result.save(save_dir=f'result/{time_str}-{img_file_path.split("/")[-1].split(".")[0]}')
        result_pic_dir = f"result/{time_str}-{img_file_path.split('/')[-1].split('.')[0]}/{img_file_path.split('/')[-1].split('.')[0]}.jpg"
        back = [{
            "text": result_text,
            "image": result_pic_dir,
        }]
        return back


model = Yolov5_Model()
# model.load_model()
# result = model.inference(img_file_path='test.jpg')  # 测试
# print(result)

# # 启动服务
server = Server(model=model)
server.server(port=8080)

# if __name__ == '__main__':
#     args = {
#         'data' : 'yolov5/data/voc_ball.yaml',
#         'weights' : 'yolov5/yolov5s.pt',
#         'cfg' : 'yolov5/models/yolov5s_ball.yaml',
#         'epochs' : 50,
#         'workers' : 1
#     }
#     model.train(**args)
