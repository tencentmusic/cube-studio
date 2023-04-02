apt install -y libgl1-mesa-glx libsm6 libxext6 g++ ffmpeg git unzip build-essential
# 安装基础环境
pip3 install  pysnooper flask numpy opencv-python Pillow requests  pydantic cpython pycocotools
# python -m pip install detectron2 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install detectron2 'git+https://gitee.com/monkeycc/detectron2.git'