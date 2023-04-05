FROM csighub.tencentyun.com/tme-kubeflow/tf_distributed_train:20201010

RUN pip3 install nni tensorflow==2.3.0 pandas sklearn scipy prettytable -i https://mirrors.tencent.com/pypi/simple

COPY job/pkgs /app/job/pkgs
COPY job/model_template/tf/*.py /app/job/model_template/tf/
COPY job/model_template/*.py /app/job/model_template/
COPY job/model_template/tf/mmoe_demo/* /app/job/model_template/tf/mmoe_demo/

WORKDIR /app
ENTRYPOINT ["python", "-m", "job.model_template.tf.mmoe_demo.driver"]