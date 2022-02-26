FROM ai.tencentmusic.com/tme-public/ubuntu-gpu:cuda10.1-cudnn7-python3.6

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

RUN pip3 freeze | xargs pip3 uninstall -y && \
    python3 -m pip install --upgrade pip && \
    pip3 install requests nni tensorflow==2.3.0 tensorflow_datasets tensorflow-addons tensorboard-plugin-profile==2.3.0 sklearn sklearn_pandas scipy gensim

COPY job/pkgs /app/job/pkgs
COPY job/tf_plain_train/*.py /app/job/tf_plain_train/

WORKDIR /app
ENTRYPOINT ["python3", "-m", "job.tf_plain_train.plain_train"]