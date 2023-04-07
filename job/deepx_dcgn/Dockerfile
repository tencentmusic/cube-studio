# Install python 3.6.
FROM python:3.6

RUN apt-get update
RUN apt-get install -y git make g++ cmake

# Install requirements
RUN pip install sklearn pandas matplotlib seaborn tqdm librosa essentia
RUN pip install xgboost

COPY job/pkgs /app/job/pkgs
COPY job/deepx_dcgn/*.py /app/job/deepx_dcgn/

ENV PYTHONPATH=/app:$PYTHONPATH
WORKDIR /app
ENTRYPOINT ["python3", "/app/job/deepx_dcgn/main.py"]