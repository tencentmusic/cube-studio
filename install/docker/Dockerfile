# docker build -t ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard:2024.04.01 -f install/docker/Dockerfile .

FROM ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard:base-python3.9-amd64-20231001

RUN wget https://docker-76009.sz.gfp.tencent-cloud.com/github/cube-studio/aihub/deeplearning/aihub.zip && mkdir -p /cube-studio/aihub && unzip aihub.zip -d /cube-studio/aihub/ && rm aihub.zip

COPY install/docker/docker-add-file/configuration.py /usr/local/lib/python3.9/dist-packages/kubernetes/client/configuration.py

COPY myapp /home/myapp/myapp
COPY myapp/static/appbuilder/frontend /data/web/frontend
COPY aihub /cube-studio/aihub
ENV PATH=/home/myapp/myapp/bin:$PATH
ENV PYTHONPATH=/home/myapp:$PYTHONPATH

COPY install/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /home/myapp/myapp/bin/myapp /entrypoint.sh

# ENTRYPOINT ["bash","/entrypoint.sh"]

# HEALTHCHECK CMD ["curl", "-f", "http://localhost:80/health"]





