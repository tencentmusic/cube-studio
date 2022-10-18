git clone https://github.com/TencentARC/GFPGAN.git
cd GFPGAN
pip install basicsr facexlib realesrgan
pip install -r requirements.txt
python setup.py develop
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models



