# docker build -t ccr.ccs.tencentyun.com/cube-studio/aihub:paddleocr  .
# docker run --name paddleocr --privileged -it -v $PWD:/app -p 8080:8080 --entrypoint='' ccr.ccs.tencentyun.com/cube-studio/aihub:paddleocr  bash
FROM ccr.ccs.tencentyun.com/cube-studio/aihub:base

# 安装基础依赖
COPY init.sh /init.sh
RUN bash /init.sh

WORKDIR /app
COPY . /app/

ENTRYPOINT ["python", "app.py"]

