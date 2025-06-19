import json
import shutil

import gradio as gr
import os,logging
import datetime
import pysnooper
import requests
import logging
import time
import cv2,os,io,base64
from fastapi import FastAPI, status, Request
from fastapi.responses import RedirectResponse
from ultralytics import YOLO
from labelstudio import LabelStudio_ML_Backend
from PIL import Image

labelstudio = LabelStudio_ML_Backend()
model_name = os.getenv('KUBEFLOW_MODEL_NAME','yolov8')
model_version = os.getenv('KUBEFLOW_MODEL_VERSION','v20240701').replace('.','')
yolo_save_dir = os.getenv('YOLO_SAVE_DIR','')  # 获取推理结果的保存地址
example = os.getenv('YOLO_EXAMPLE','')  # 获取推理结果的保存地址
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.getenv('MODELPATH',os.getenv('KUBEFLOW_MODEL_PATH','/yolov8/yolov8n.pt'))
# https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
device='cpu'

# 加载模型
# 这里是添加的gpu识别
resource_gpu = os.getenv('RESOURCE_GPU', '')
resource_gpu = resource_gpu.split('(')[0]
resource_gpu = resource_gpu.split(',')[-1]
resource_gpu = float(resource_gpu) if resource_gpu else 0
if resource_gpu>0:
    device = 'cuda:0'

# 下载模型
if 'http://' in model_path or "https://" in model_path:
    model_path = 'best.pt'
    file = open(model_path,mode='w')
    res = requests.get(model_path)
    if res.status_code == 200:
        file.write(res.content)
        file.close()

model = YOLO(model_path)

# 创建 FastAPI 实例
app = FastAPI()

@pysnooper.snoop(watch_explode=('boxes'))
def inference(source,return_type='image'):
    if 'http://' in source or 'https://' in source:
        response = requests.get(source)
        ext = source[source.rindex(".") + 1:]
        ext = ext if len(ext) < 6 else 'jpg'
        source = f'input-{int(time.time() * 1000)}.{ext}'

        if os.path.exists(source):
            os.remove(source)


        # 确保请求成功
        if response.status_code == 200:
            # 将视频内容写入本地文件
            with open(source, "wb") as file:
                file.write(response.content)
                print(f"文件已成功保存到: {source}")
        else:
            print(f"请求失败，状态码: {response.status_code}")

    results = model(source,device=device)
    result = None
    if results:
        result = results[0]
    if not result:
        if return_type == 'image':
            return source
        if return_type == 'box_json':
            return {}
        if return_type == 'all_json':
            return {}
        return None

    if return_type=='image':

        save_dir = 'results'
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'{int(time.time() * 1000)}.jpg')  # img.jpg
        im_array = result.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save(save_path)  # save image

        return save_path
    elif return_type=='box_json':
        names = result.names
        boxes = result.boxes
        orig_height,orig_width = boxes.orig_shape[0],boxes.orig_shape[1]
        xywhns = boxes.xywhn.tolist()
        xywhns = [[round(box[0],6),round(box[1],6),round(box[2],6),round(box[3],6)] for box in xywhns]
        # 把类型序号换成label名称
        cls = boxes.cls.tolist()
        cls = [names[index] for index in cls]
        conf = boxes.conf.tolist()
        return {
            "names":list(names.values()),
            "labels":cls,
            "scores":conf,
            "xywhns":xywhns,
            "orig_shape":[orig_width,orig_height]
        }
    elif return_type=='all_json':
        # 准备存储结果的列表
        output_data = []
        # 遍历每个结果
        for result in results:
            if hasattr(result, 'masks'):
                for mask in result.masks:
                    output_data.append({
                        'class': mask.cls,
                        'confidence': mask.conf,
                        'segmentation': mask.segmentation.tolist()  # 将numpy数组转换为列表
                    })

            # 获取检测框、类别和置信度
            if hasattr(result, 'boxes'):
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
                scores = result.boxes.conf.cpu().numpy()  # 获取置信度分数
                labels = result.boxes.cls.cpu().numpy()  # 获取类别标签

                for box, score, label in zip(boxes, scores, labels):
                    json_result = {
                        'box': box.tolist(),
                        'score': float(score),
                        'label': int(label)
                    }
                    output_data.append(json_result)

        return {
            "result":output_data
        }



label = '一站式机器学习平台yolov8目标识别推理服务'
describe = '云原生一站式机器学习/深度学习AI平台，支持sso登录，多租户/多项目组，数据资产对接，notebook在线开发，拖拉拽任务流pipeline编排，多机多卡分布式算法训练，超参搜索，推理服务VGPU，多集群调度，边缘计算，serverless，标注平台，自动化标注，数据集管理，大模型一键微调，llmops，私有知识库，AI应用商店，支持模型一键开发/推理/微调，私有化部署，支持国产cpu/gpu/npu芯片，支持RDMA，支持pytorch/tf/mxnet/deepspeed/paddle/colossalai/horovod/spark/ray/volcano分布式'
gradio_examples=[
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000000597.jpg",
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000000797.jpg",
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000000897.jpg",
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000001397.jpg",
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000001497.jpg",
    "https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/media-download/train2014/COCO_train2014_000000001697.jpg"
] if not example else example.split(',')

with gr.Blocks(title=label,theme=gr.themes.Default(text_size='lg')) as demo:

    with gr.Row():
        html=f'<h1 style="text-align: center; margin-bottom: 1rem">{label}</h1>'
        title = gr.HTML(value = html)
    # 介绍
    with gr.Row():
        description = gr.Markdown(value = describe)
    with gr.Row():
        with gr.Tab("推理"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 输入
                    inputs = gr.Image(value=None, label='待推理图片',type="filepath")
                    submit_button = gr.Button("提交")
                with gr.Column(scale=1):
                    outputs = gr.Image(label='图片输出结果')
                submit_button.click(inference, inputs=inputs,outputs=outputs)
            with gr.Row():
                # 遍历多个示例
                gr.Examples(
                    examples=gradio_examples,
                    inputs=inputs,
                    outputs=outputs,
                    fn=inference,
                    cache_examples=False,
                )
            # with gr.Row():
            #     path = os.path.join(current_dir,'gradio_rec.txt')
            #     if os.path.exists(path):
            #         choices = open(path).readlines()
            #         choices = [x.strip() for x in choices if x.strip()]
            #         gr.Gallery(value=choices,label='其他应用',show_label=False,allow_preview=False)
        with gr.Tab("接口"):
            api_markdown = ''
            if os.path.exists(os.path.join(current_dir, 'yolov8_api.md')):
                api_markdown = open(os.path.join(current_dir, 'yolov8_api.md')).read().strip()
                api_markdown = api_markdown.replace('MODEL_NAME',model_name).replace('MODEL_VERSION',model_version)
            jiekou = gr.Markdown(value=api_markdown)


# @pysnooper.snoop()
def labelstudio_download_image(image_path,save_dir='result',**kwargs):
    headers = {
        "Authorization": f"Token {labelstudio.access_token}"
    }
    save_path = os.path.join(save_dir,f'result{int(time.time() * 1000)}.jpg')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        os.remove(save_path)
    if 'http://' in image_path or "https://" in image_path:
        if labelstudio.hostname in image_path:
            import requests
            from urllib import parse
            from urllib.parse import urlparse
            res = urlparse(image_path)
            params = parse.parse_qs(res.query)
            d = params.get('d',[''])[0]
            image_path=f'http://{res.netloc}/static/'+d
            file = open(save_path,mode='wb')
            print(image_path,headers)
            # res = requests.get(image_path,headers=headers)
            res = requests.get(image_path)
            if res.status_code==200:
                file.write(res.content)
                file.close()
                image_path=save_path
        else:
            import requests
            # 发送请求并获取图片内容
            response = requests.get(image_path)
            # 确保请求成功
            if response.status_code == 200:
                # 将视频内容写入本地文件
                with open(save_path, "wb") as file:
                    file.write(response.content)
                    print(f"图片已成功保存到: {save_path}")
            else:
                print(f"请求失败，状态码: {response.status_code}")
            image_path = save_path

    if '/labelstudio/data/upload' in image_path:
        image_path = image_path.replace('/labelstudio/data/upload','/labelstudio/media/upload')

    return image_path



@app.post("/demo/predict")
async def demo_predict(request: Request):

    global all_predict, step_num
    data = await request.json()

    image_data = data['image']
    # 解码图像
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    file_name = f'{int(time.time())}'
    image_save_path = f'input/{file_name}.jpg'

    os.makedirs('input', exist_ok=True)
    image.save(image_save_path)
    # 推理预测
    result = inference(image_save_path, return_type='all_json')

    return result

@app.post("/labelstudio/predict")
async def labelstudio_predict(request: Request):
    data = await request.json()
    tasks, label_config = data.get('tasks',[]),data.get('label_config','')
    back = []
    for task in tasks:
        image_path = task['data']['image']  # 获取图片路径
        image_path = labelstudio_download_image(image_path)
        predictions = inference(image_path,return_type='box_json')
        result = []
        labels = predictions['labels']
        scores = predictions['scores']
        xywhns =predictions['xywhns']
        height, width = predictions['orig_shape'][0],predictions['orig_shape'][1]

        import xml.etree.ElementTree as ET
        root = ET.fromstring(label_config)
        choices = []
        # 遍历所有的item节点
        for item in root.findall('.//Label'):  # 这里'.//item'是XPath，用于查找所有的item节点
            # 获取指定属性
            attribute_value = item.get('value')  # 这里'attribute'是你想要获取的属性名
            print(attribute_value)
            if attribute_value:
                choices.append(attribute_value)
        choices = list(set(choices))
        choices_lower = [x.lower() for x in choices]

        for index, label in enumerate(labels):
            if label.lower() not in choices_lower:
                continue
            label = choices[choices_lower.index(label.lower())]

            xywhn = xywhns[index]
            # 换成coco数据集格式 start_x,start_y,width,height
            xywhn = [int((xywhn[0] - xywhn[2] / 2) * 100), int((xywhn[1] - xywhn[3] / 2) * 100), int(xywhn[2] * 100),int(xywhn[3] * 100)]
            score = scores[index]
            result.append({
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "x": xywhn[0],
                    "y": xywhn[1],
                    "width": xywhn[2],
                    "height": xywhn[3],
                    "rotation": 0,
                    "rectanglelabels": [label]
                },
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "origin": "prediction",
                "score":int(score*100)
            })

        back.append({
            'result': result,
            # optionally you can include prediction scores that you can use to sort the tasks and do active learning
            'score': int(sum(scores) / max(len(scores), 1)),
            'model_version': f'{model_name}-{model_version}'
        })

    response = {
        'results': back,
        'model_version': f'{model_name}-{model_version}'
    }
    # print(response)
    return response

# 添加新的路由
@app.post("/labelstudio/setup")
async def labelstudio_setup(request: Request):
    data = await request.json()
    return labelstudio.labelstudio_setup(**data)

# 添加新的路由
@app.get("/labelstudio/health")
async def labelstudio_health():
    return {}

# 添加新的路由
@app.get("/")
async def index():
    return RedirectResponse(url="/gradio", status_code=status.HTTP_302_FOUND)
all_predict=[]
YOLO_SAVE_STEP=int(os.getenv('YOLO_SAVE_STEP','1'))   # 每推理多少步保存一次
step_num=0
# define your API endpoint
@app.post(f'/v1/models/{model_name}/versions/{model_version}/predict')
async def predict(request: Request):
    global all_predict,step_num
    data = await request.json()

    image_data = data['image']
    # 解码图像
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    file_name = f'{int(time.time())}'
    image_save_path = f'input/{file_name}.jpg'
    if yolo_save_dir:
        save_dir = os.path.join(yolo_save_dir,model_name,model_version)
        image_save_dir = os.path.join(save_dir,"images")
        label_save_dir = os.path.join(save_dir, "labels")
        os.makedirs(image_save_dir,exist_ok=True)
        os.makedirs(label_save_dir,exist_ok=True)
        image_save_path = os.path.join(image_save_dir,f'{file_name}.jpg')


    os.makedirs('input',exist_ok=True)
    image.save(image_save_path)
    # 推理预测
    result = inference(image_save_path,return_type='box_json')

    if result:
        if yolo_save_dir:
            step_num+=1
            with pysnooper.snoop():
                names = result['names']
                labels = result['labels']
                scores = result['scores']
                xywhns = result['xywhns']
                # 生成labels
                label_save_path = os.path.join(label_save_dir,f'{file_name}.txt')
                label_save_path_file=open(label_save_path,mode='w')
                for index,label in enumerate(labels):
                    box = xywhns[index]
                    box=[names.index(label),box[0],box[1],box[2],box[3]]
                    box = [str(x) for x in box]
                    label_save_path_file.write(' '.join(box)+"\n")
                label_save_path_file.close()
                # 生成data.yaml
                data_yaml_path = os.path.join(save_dir,'data.yaml')
                if not os.path.exists(data_yaml_path):
                    json_data = {"names": {}}
                    for index,name in enumerate(names):
                        json_data['names'][index]=name
                    # 将 Python 字典转换为 YAML 格式
                    import yaml
                    yaml_data = yaml.dump(json_data, allow_unicode=True, default_flow_style=False)
                    # 将 YAML 数据写入到文件
                    with open(data_yaml_path, 'w', encoding='utf-8') as yaml_file:
                        yaml_file.write(yaml_data)

                # 生成train.txt
                train_save_path_file=open(os.path.join(save_dir,'train.txt'),mode='w')
                train_save_path_file.write(f'images/{file_name}.jpg')
                train_save_path_file.close()
                # 生成标注平台json格式
                # if '/mnt' in image_save_path:
                #     image_save_path =

                height, width = result['orig_shape'][0], result['orig_shape'][1]
                label_studio_data=[]
                for index, label in enumerate(labels):

                    xywhn = xywhns[index]
                    # 换成coco数据集格式 start_x,start_y,width,height
                    xywhn = [int((xywhn[0] - xywhn[2] / 2) * 100), int((xywhn[1] - xywhn[3] / 2) * 100),int(xywhn[2] * 100),int(xywhn[3] * 100)]
                    score = scores[index]
                    label_studio_data.append({
                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,
                        "value": {
                            "x": xywhn[0],
                            "y": xywhn[1],
                            "width": xywhn[2],
                            "height": xywhn[3],
                            "rotation": 0,
                            "rectanglelabels": [label]
                        },
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "origin": "prediction",
                        "score": int(score * 100)
                    })
                #
                if not all_predict and os.path.exists(os.path.join(save_dir,'labelstudio.json')):
                    all_predict = json.load(open(os.path.join(save_dir,'labelstudio.json')))

                all_predict.append({
                    "data":{
                        "image":("/labelstudio/data/local-files/?d="+image_save_path.strip('/')) if '/mnt' in image_save_path else image_save_path,
                    },
                    "annotations":[
                        {
                            "result":label_studio_data
                        }
                    ]
                })
                # 每n次进行一次保存
                if step_num>=YOLO_SAVE_STEP:
                    json.dump(all_predict,open(os.path.join(save_dir,'labelstudio.json'),mode='w'),indent=4,ensure_ascii=False)
                    step_num=0

        else:
            os.remove(image_save_path)
        return result
    else:
        return {
            "labels":[],
            "scores":[],
            "xywhs":[],
            "orig_shape":[]
        }

app = gr.mount_gradio_app(app, demo, path="/gradio")

# 启动 FastAPI 应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)









