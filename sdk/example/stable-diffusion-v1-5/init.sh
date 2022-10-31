# 初始化环境
git clone https://github.com/runwayml/stable-diffusion.git

cd stable-diffusion/
conda env create -f environment.yaml
# 激活环境
source activate
# 退出环境
source deactivate
conda activate ldm

pip3 install pysnooper flask requests huggingface_hub Pillow
# huggingface 登录
pip install huggingface_hub
huggingface-cli login
git config --global credential.helper store
# 下载模型
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -P /stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt

git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
pip3 install --upgrade diffusers transformers scipy

