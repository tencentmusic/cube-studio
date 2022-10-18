
# docker build -t ccr.ccs.tencentyun.com/cube-studio/gfpgan:deoldify  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base

# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
RUN bash /init.sh

# 安装文件
WORKDIR /app
COPY * /app/


ENTRYPOINT ["python", "app.py"]

# docker run --name gfpgan --rm --privileged -it -v $PWD:/app -p 8080:8080 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/aihub:gfpgan
