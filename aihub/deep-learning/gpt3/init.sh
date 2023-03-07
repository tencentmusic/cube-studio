pip3 install torch torchvision torchaudio
pip install modelscope
pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install deepspeed
conda install -y -c conda-forge tensorboard
cd /app/ && python app.py download_model

