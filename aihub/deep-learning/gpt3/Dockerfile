
# 构建应用镜像 docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:gpt3  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base-cuda11.4-python3.9
# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
COPY * /app/

ENV TORCH_CUDA_ARCH_LIST "compute capability"
RUN pip3 install torch torchvision torchaudio modelscope deepspeed Flask==2.1.3
RUN pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

RUN conda install -y -c nvidia cuda-nvcc
RUN conda install -y -c conda-forge tensorboard
RUN cd /app/ && python download_model.py

# RUN bash /init.sh
# 安装文件
WORKDIR /app
#ENTRYPOINT ["python", "app.py"]

