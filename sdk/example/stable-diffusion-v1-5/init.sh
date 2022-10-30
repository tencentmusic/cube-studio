git clone https://github.com/runwayml/stable-diffusion.git

conda install -y pytorch torchvision -c pytorch
pip3 install pysnooper flask requests Pillow torch torchvision opencv-python
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
pip3 install --upgrade diffusers transformers scipy
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
huggingface-cli login

