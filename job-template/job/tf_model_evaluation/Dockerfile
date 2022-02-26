FROM tensorflow/tensorflow:2.3.0

RUN pip uninstall -y protobuf && \
    pip install protobuf  nni  tensorflow_datasets  sklearn  sklearn_pandas scipy  gensim  prettytable

COPY job/pkgs /app/job/pkgs
COPY job/model_template /app/job/model_template
COPY job/tf_model_evaluation/*.py /app/job/tf_model_evaluation/

WORKDIR /app
ENTRYPOINT ["python", "-m", "job.tf_model_evaluation.model_evaluation"]