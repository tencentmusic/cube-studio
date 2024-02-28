# docker build -t ccr.ccs.tencentyun.com/cube-studio/kubeflow-dashboard-frontend:2024.03.01 -f install/docker/dockerFrontend/Dockerfile .

FROM nginx

MAINTAINER star-frontend

# 环境变量
ENV TZ=Asia/Shanghai \
    RUN_USER=nginx \
    RUN_GROUP=nginx \
    DATA_DIR=/data/web \
    LOG_DIR=/data/log/nginx

# 工作目录
WORKDIR ${DATA_DIR}

# 切换为上海时区
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# 创建日志文件夹
RUN mkdir ${LOG_DIR} -p
RUN chown nginx.nginx -R ${LOG_DIR}

RUN mkdir ./frontend

# 拷贝dist包文件
COPY ./myapp/static/appbuilder/frontend ./frontend
RUN chmod -R 777 frontend
# 拷贝nginx配置文件
ADD /install/docker/dockerFrontend/nginx.conf /etc/nginx/nginx.conf
ADD /install/docker/dockerFrontend/nginx.80.conf /etc/nginx/conf.d/default.conf
ADD /install/docker/dockerFrontend/start.sh /start.sh
ENTRYPOINT nginx -g "daemon off;"
#ENTRYPOINT ["sh", "/start.sh"]
