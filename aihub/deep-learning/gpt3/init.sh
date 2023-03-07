export TORCH_CUDA_ARCH_LIST="compute capability"
pip3 install torch torchvision torchaudio
pip install modelscope
pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install deepspeed
#conda install -y -c nvidia cuda-nvcc
#conda install -y -c conda-forge tensorboard
pip install tensorboard
cd /app/ && python download_model.py

