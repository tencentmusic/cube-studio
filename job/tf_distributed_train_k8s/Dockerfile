FROM python:3.7.8

RUN wget https://github.com/wercker/stern/releases/download/1.11.0/stern_linux_amd64 && chmod +x stern_linux_amd64 &&  mv stern_linux_amd64 /usr/bin/stern
RUN apt update && apt install -y --force-yes --no-install-recommends vim apt-transport-https gnupg2 ca-certificates-java rsync jq  wget git dnsutils iputils-ping net-tools curl locales zip software-properties-common && apt-get autoclean

RUN python -m pip install --upgrade pip && \
    pip install requests -i https://mirrors.tencent.com/pypi/simple && \
    pip install kubernetes -i https://mirrors.tencent.com/pypi/simple && pip install psutil pysnooper

COPY job/pkgs /app/job/pkgs
COPY job/tf_distributed_train/*.py /app/job/tf_distributed_train/
COPY job/pkgs/k8s/*kubeconfig /root/.kube/

WORKDIR /app
ENTRYPOINT ["python", "-m", "job.tf_distributed_train.tfjob_launcher"]