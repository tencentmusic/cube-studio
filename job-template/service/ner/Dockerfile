FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime


COPY . /SVC

RUN python3 -m pip install pickle-mixin -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install FastAPI -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple


ENTRYPOINT ["python3", "/SVC/launcher.py"]
