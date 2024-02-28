# 配置国内源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
# 安装基础包
pip install  pysnooper psutil requests numpy  pyinstaller argparse pika Pillow torch==2.0.1 torchvision==0.15.2
# 下载模型
if [ "$VC_TASK_INDEX" == "0" ] && [ ! -f "./resnet50.pth" ]; then
  wget -O resnet50.pth https://cube-studio.oss-cn-hangzhou.aliyuncs.com/pipeline/offline-inference/resnet50.pth
fi
# 执行离线推理
python demo.py

