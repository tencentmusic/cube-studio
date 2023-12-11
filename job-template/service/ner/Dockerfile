FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime


COPY . /SVC

RUN python3 -m pip install pickle-mixin -i https://mirrors.aliyun.com/pypi/simple
RUN python3 -m pip install FastAPI -i https://mirrors.aliyun.com/pypi/simple
RUN python3 -m pip install uvicorn -i https://mirrors.aliyun.com/pypi/simple


ENTRYPOINT ["python3", "/SVC/launcher.py"]
