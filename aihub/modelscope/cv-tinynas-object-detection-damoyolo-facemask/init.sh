
# 安装基础环境
apt install -y libgl1-mesa-glx libsm6 libxext6 g++ build-essential
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install pysnooper flask numpy opencv-python Pillow requests torch torchvision
