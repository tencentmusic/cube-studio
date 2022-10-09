FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime


RUN python3 -m pip install uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python3 -m pip install fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python3 -m pip install python-multipart  -i https://pypi.tuna.tsinghua.edu.cn/simple


WORKDIR /workspace
RUN chmod -R a+w /workspace
ADD ./launcher.py /workspace/

ENTRYPOINT ["python3", "launcher.py"]