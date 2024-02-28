FROM ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9
COPY sources.list /etc/apt/sources.list
RUN apt update -y
RUN git clone https://github.com/WongKinYiu/yolov7.git
WORKDIR /yolov7
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN apt-get install -y python3-opencv && pip install -r requirements.txt
RUN pip install gradio==4.15.0 uvicorn --index-url https://mirrors.aliyun.com/pypi/simple
# 下载预训练模型
RUN mkdir -p weights dataset && cd weights && wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt && wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
# 拷贝配置文件
COPY . /yolov7/

RUN chmod 777 launcher.sh
ENTRYPOINT ["bash","launcher.sh"]

# 测试预训练模型推理
# python detect.py --weights weights/yolov7.pt --source inference/images

# 测试微调
# python train.py --weights weights/yolov7_training.pt --cfg /mnt/admin/coco/yolov7.yaml --data /mnt/admin/coco/coco.yaml --device cpu --batch-size 64 --epoch 1

# 微调后推理
#python detect.py --weights runs/train/exp2/weights/best.pt --source inference/images



