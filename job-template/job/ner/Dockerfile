FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime


COPY . /NER

RUN python3 -m pip install sklearn_crfsuite -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install minio -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT ["python3", "/NER/main4argo.py"]