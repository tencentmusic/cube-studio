apt install -y build-essential

conda install -y -c conda-forge sox libsndfile swig bzip2 libflac bc
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
pip install paddlespeech -i https://pypi.tuna.tsinghua.edu.cn/simple

wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav -P /
wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav -P /