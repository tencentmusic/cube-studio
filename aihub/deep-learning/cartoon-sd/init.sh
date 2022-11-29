wget -c https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/naifu.tar
tar -xvf naifu.tar && rm naifu.tar
pip install torch torchvision torchaudio dotmap fastapi uvicorn omegaconf transformers sentence_transformers faiss-cpu einops pytorch_lightning ftfy scikit-image torchdiffeq jsonmerge -i https://mirror.baidu.com/pypi/simple