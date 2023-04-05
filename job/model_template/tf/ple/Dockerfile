FROM csighub.tencentyun.com/tme-kubeflow/tf_distributed_train:20201010

RUN pip3 install nni tensorflow==2.3.0 tensorboard-plugin-profile==2.3.0 tensorflow-addons pandas sklearn scipy==1.4.1 prettytable tqdm -i https://mirrors.tencent.com/pypi/simple

COPY job/pkgs /app/job/pkgs
COPY job/tf_model_evaluation/*.py /app/job/tf_model_evaluation/
COPY job/tf_model_offline_predict/*.py /app/job/tf_model_offline_predict/
COPY job/tf_keras_train/*.py /app/job/tf_keras_train/
COPY job/tf_distributed_evaluation/*.py /app/job/tf_distributed_evaluation/
COPY job/model_template/tf/*.py /app/job/model_template/tf/
COPY job/model_template/*.py /app/job/model_template/
COPY job/model_template/tf/ple/* /app/job/model_template/tf/ple/
COPY job/model_template/tf/ple/custom/*.py /app/job/model_template/tf/ple/custom/

WORKDIR /app
ENTRYPOINT ["python", "-m", "job.model_template.tf.ple.driver"]