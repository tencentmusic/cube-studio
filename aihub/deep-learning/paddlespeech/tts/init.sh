apt install build-essential
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech

conda install -y -c conda-forge sox libsndfile swig bzip2 libflac bc
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
pip install -e .
