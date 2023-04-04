FROM csighub.tencentyun.com/tme-kubeflow/tlinux2.2:strong

RUN pip install requests pyDes pyfastcopy -i https://mirrors.tencent.com/pypi/simple

COPY job/pkgs /app/job/pkgs
COPY job/tesla_job_handler/*.py /app/job/tesla_job_handler/

ENV PYTHONPATH=/app:$PYTHONPATH
WORKDIR /app
ENTRYPOINT ["python3", "-m", "job.tesla_job_handler.main"]


