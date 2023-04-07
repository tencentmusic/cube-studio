# Install python 3.6.
FROM python:3.6

RUN apt-get update
RUN apt-get install -y git make g++ cmake

# Install requirements
RUN pip install sklearn pandas matplotlib seaborn tqdm librosa essentia
RUN pip install tensorflow==2.3.0
COPY job/pkgs /app/job/pkgs
COPY job/youtube_dnn/*.py /app/

ENV PYTHONPATH=/app:/app/job/pkgs:$PYTHONPATH
WORKDIR /app
ENTRYPOINT ["python3", "main.py"]