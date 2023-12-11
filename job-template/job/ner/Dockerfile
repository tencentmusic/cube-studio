FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime


COPY . /NER

RUN python3 -m pip install sklearn_crfsuite -i https://mirrors.aliyun.com/pypi/simple
RUN python3 -m pip install minio -i https://mirrors.aliyun.com/pypi/simple
RUN python3 -m pip install tqdm -i https://mirrors.aliyun.com/pypi/simple
RUN python3 -m pip install transformers -i https://mirrors.aliyun.com/pypi/simple

ENTRYPOINT ["python3", "/NER/main4argo.py"]