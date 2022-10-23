
# 构建应用镜像 docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:app1  .
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base
# 安装基础环境
WORKDIR /
COPY init.sh /init.sh
RUN bash /init.sh
# 安装文件
WORKDIR /app
COPY * /app/
ENTRYPOINT ["python", "app.py"]
# 应用开发
# docker run --name app1 --privileged -it -v $PWD:/app -p 8080:8080 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/aihub:app1 bash
# 应用体验
# docker run --name app1 --rm --privileged -it -p 8080:8080 ccr.ccs.tencentyun.com/cube-studio/aihub:app1