FROM ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda11.8.0-cudnn8-python3.9
#COPY sources.list /etc/apt/sources.list
RUN apt update -y
WORKDIR /yolov8

RUN apt-get install -y python3.9-dev python3-opencv
RUN pip install torch==2.5.1  torchvision==0.20.1  ultralytics==8.2.55

RUN pip install gradio==4.43.0 uvicorn aiohttp pyyaml --index-url https://mirrors.aliyun.com/pypi/simple
# 下载预训练模型
RUN wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt && wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt && wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt && wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt && wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt
# 下载其他
RUN wget -P /root/.config/Ultralytics/ https://ultralytics.com/assets/Arial.ttf
# 拷贝配置文件
COPY . /yolov8/

# 安装aihub包
ENV TZ=Asia/Shanghai
ENV PYTHONPATH=/src/:/github/
ENV NODE_PATH=$NODE_HOME/lib/node_modules:${PATH:-}
ENV PATH=$NODE_HOME/bin:${PATH:-}

RUN rm -rf /usr/bin/python3 && ln -s /usr/bin/python /usr/bin/python3

ENTRYPOINT ["python","train.py"]



