# docker build -t ai.tencentmusic.com/tme-public/kubeflow-dashboard:2021.11.01 -f install/docker/Dockerfile .

FROM ai.tencentmusic.com/tme-public/kubeflow-dashboard:base

RUN apt-get update && apt-get install -y mysql-client zip

RUN curl -sL https://deb.nodesource.com/setup_14.x | bash
RUN apt-get -y install nodejs && npm install yarn -g

RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python && rm /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install docstring_parser kfp kfserving
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl && mv kubectl /usr/bin/
RUN mkdir /kubeflow && wget -P /kubeflow/ https://docker-76009.sz.gfp.tencent-cloud.com/kubeflow/src.tar.gz && tar -zxvf /kubeflow/src.tar.gz -C /kubeflow/ && rm /kubeflow/src.tar.gz
# katib 需要kubernetes10.0版本
#RUN pip install kubernetes==10.0.0
# kfserving需要kubernetes 12.0版本
RUN pip install kubernetes==12.0.1 click==6.7
COPY install/docker/docker-add-file/rest.py /usr/local/lib/python3.6/dist-packages/kubernetes/client/rest.py
COPY install/docker/docker-add-file/__init__.py /usr/local/lib/python3.6/dist-packages/marshmallow_enum/__init__.py

COPY myapp /home/myapp/myapp
ENV PATH=/home/myapp/myapp/bin:$PATH
ENV PYTHONPATH=/home/myapp:/kubeflow/src/pipelines/sdk/python/:/kubeflow/src/fairing/:/kubeflow/src/kfserving/python/kfserving/:/kubeflow/src/tf-operator/sdk/python/:/kubeflow/src/xgboost-operator/sdk/python/:/kubeflow/src/katib/sdk/python/v1alpha3/:$PYTHONPATH

USER root
COPY install/docker/entrypoint.sh /entrypoint.sh

# ENTRYPOINT ["/entrypoint.sh"]
# HEALTHCHECK CMD ["curl", "-f", "http://localhost:80/health"]






