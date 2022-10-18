
# docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:deoldify  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base

# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
RUN bash /init.sh
ENV PYTHONPATH=/DeOldify:$PYTHONPATH
# 安装文件
WORKDIR /app
COPY * /app/

ENTRYPOINT ["python", "app.py"]

# docker run --name deoldify --rm --privileged -it -v $PWD:/app -p 8080:8080 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/aihub:deoldify

