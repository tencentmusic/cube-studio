FROM tensorflow/tensorflow:2.3.0

RUN pip uninstall -y protobuf && \
    pip install protobuf -i https://mirrors.tencent.com/pypi/simple && \
    pip install nni -i https://mirrors.tencent.com/pypi/simple && \
    pip install tensorflow_datasets -i https://mirrors.tencent.com/pypi/simple && \
    pip install sklearn -i https://mirrors.tencent.com/pypi/simple && \
    pip install sklearn_pandas -i https://mirrors.tencent.com/pypi/simple && \
    pip install scipy -i https://mirrors.tencent.com/pypi/simple && \
    pip install gensim -i https://mirrors.tencent.com/pypi/simple && \
    pip install prettytable -i https://mirrors.tencent.com/pypi/simple

COPY job/pkgs /app/job/pkgs
COPY job/tf_model_offline_predict/*.py /app/job/tf_model_offline_predict/

WORKDIR /app
ENTRYPOINT ["python", "-m", "job.tf_model_offline_predict.main"]