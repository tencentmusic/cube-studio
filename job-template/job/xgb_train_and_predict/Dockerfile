FROM python:3.6
USER root

WORKDIR /app
RUN apt-get update
RUN apt -y install g++ cmake

RUN pip3 install xgboost pandas numpy joblib sklearn

COPY job/xgb_train_and_predict/* /app/
COPY job/pkgs /app/job/pkgs
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python3", "launcher.py"]
