apt install -y build-essential
git clone https://github.com/JLWLL/PaddleDetection.git
cd PaddleDetection || exit
python -m pip install paddlepaddle==2.4.2 -i https://mirror.baidu.com/pypi/simple
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
python setup.py install
wget https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams