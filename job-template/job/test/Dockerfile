# docker buildx build --platform linux/amd64,linux/arm64  -t ccr.ccs.tencentyun.com/cube-studio/python:ml . --push

# 使用官方Python镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和模型文件
COPY . .


# 暴露端口
EXPOSE 5000

# 运行应用
CMD ["python", "train.py"]
