git clone https://github.com/runwayml/stable-diffusion.git

cd stable-diffusion/
conda env create -f environment.yaml
# 激活环境
source activate
# 退出环境
source deactivate
conda activate ldm

pip3 install pysnooper flask requests huggingface_hub

pip install huggingface_hub
huggingface-cli login
git config --global credential.helper store
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt

git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
pip3 install --upgrade diffusers transformers scipy

