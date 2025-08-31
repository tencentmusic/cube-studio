
# docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/colossalai:cuda-11.3-rdma-python3.9-pytorch1.12-colossalai-0.3.1 .

FROM hpcaitech/cuda-conda:11.3

# enable passwordless ssh
RUN mkdir ~/.ssh && \
    printf "Host * \n    ForwardAgent yes\nHost *\n    StrictHostKeyChecking no" > ~/.ssh/config && \
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# enable RDMA support
RUN apt-get update && \
    apt-get install -y infiniband-diags perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/ && conda config --show channels
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple


# install torch
RUN conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# install ninja
RUN apt-get update && \
    apt-get install -y --no-install-recommends ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install apex，网络不好，本地下载再传进去
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout 91fcaa && \
    pip install packaging && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" ./

# install colossalai
ARG VERSION=v0.3.1
RUN git clone -b ${VERSION} https://github.com/hpcaitech/ColossalAI.git \
    && cd ./ColossalAI \
    && CUDA_EXT=1 pip install -v --no-cache-dir .

RUN cd ./ColossalAI/requirements/ && pip install -r requirements.txt

# install titans
RUN pip install --no-cache-dir titans

# install tensornvme
RUN conda install -y cmake && \
    git clone https://github.com/hpcaitech/TensorNVMe.git && \
    cd TensorNVMe && \
    apt update -y && apt install -y libaio-dev && \
    pip install -r requirements.txt && \
    pip install -v --no-cache-dir .
# 安装基础包
RUN pip install pytest datasets numpy transformers decorator tensorboard sentencepiece numpy tqdm psutil packaging pre-commit rich click fabric contexttimer ninja torch>=1.13.1 safetensors einops
