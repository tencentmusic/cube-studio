pip3 install pysnooper flask requests Pillow torch==1.11.0 torchvision==0.12.0 albumentations==0.4.3 opencv-python==4.6.0.66 pudb==2019.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.4.2 omegaconf==2.1.1 test-tube streamlit einops==0.3.0 torch-fidelity==0.3.0 transformers==4.19.2 torchmetrics==0.6.0 kornia==0.6 imWatermark
pip3 install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers -e git+https://github.com/openai/CLIP.git@main#egg=clip


# 下载模型
wget https://docker-76009.sz.gfp.tencent-cloud.com/github/cube-studio/aihub/deeplearning/stable-diffusion/v1-5-pruned-emaonly.ckpt -O /model.ckpt

git clone https://github.com/CompVis/stable-diffusion.git
cd /stable-diffusion
pip install -e .
mkdir -p /stable-diffusion/models/ldm/stable-diffusion-v1/
ln -s model.ckpt /stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt
cd /
git clone https://github.com/CompVis/taming-transformers.git
cd /taming-transformers
pip install -e .


