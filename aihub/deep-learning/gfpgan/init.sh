git clone https://github.com/TencentARC/GFPGAN.git
cd GFPGAN
pip install basicsr facexlib realesrgan
pip install -r requirements.txt
python setup.py develop
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models/
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P experiments/pretrained_models/

wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -P /GFPGAN/gfpgan/weights/
wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -P /GFPGAN/gfpgan/weights/

