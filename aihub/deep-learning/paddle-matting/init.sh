apt install -y build-essential
git clone https://gitee.com/paddlepaddle/PaddleSeg.git
cd PaddleSeg || exit
python -m pip install paddlepaddle==2.4.2 -i https://mirror.baidu.com/pypi/simple
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
pip install -e .
cd Matting || exit
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
cd / || exit
git clone https://gitee.com/JLWL1945709919/Matting.git
cd /Matting || exit
wget https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_512.pdparams
