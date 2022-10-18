
# docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:deoldify  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base

# 安装基础依赖
WORKDIR /app
COPY * /app/
RUN bash /app/init.sh

ENTRYPOINT ["python", "app.py"]

# docker run --name deoldify --rm --privileged -it -v $PWD:/app -p 8080:8080 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/aihub:deoldify
