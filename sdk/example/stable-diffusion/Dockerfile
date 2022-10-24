
# 构建应用镜像 docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:stable-diffusion  .
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
# docker run --name stable-diffusion --privileged -it -v $PWD:/app -p 8080:8080 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/aihub:stable-diffusion bash
# 应用体验
# docker run --name stable-diffusion --rm --privileged -it -p 8080:8080 ccr.ccs.tencentyun.com/cube-studio/aihub:stable-diffusion