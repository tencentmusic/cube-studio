# docker build -t csighub.tencentyun.com/tme-kubeflow/gpu-tf2.1-dist-demo .
FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install tensorflow_datasets

COPY multi_worker_strategy-with-keras.py /
ENTRYPOINT ["python", "/multi_worker_strategy-with-keras.py", "--saved_model_dir", "/train/saved_model/", "--checkpoint_dir", "/train/checkpoint"]



