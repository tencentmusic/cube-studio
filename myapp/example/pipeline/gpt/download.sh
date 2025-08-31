
pip3 install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT="https://hf-mirror.com"

# 下载baichuan2
mkdir -p baichuan-inc/Baichuan2-7B-Chat
huggingface-cli download --resume-download baichuan-inc/Baichuan2-7B-Chat --local-dir baichuan-inc/Baichuan2-7B-Chat --local-dir-use-symlinks False

mkdir -p baichuan-inc/Baichuan2-13B-Chat
huggingface-cli download --resume-download baichuan-inc/Baichuan2-13B-Chat --local-dir baichuan-inc/Baichuan2-13B-Chat --local-dir-use-symlinks False

# 下载chatglm2
mkdir -p THUDM/chatglm2-6b
huggingface-cli download --resume-download THUDM/chatglm2-6b --local-dir THUDM/chatglm2-6b --local-dir-use-symlinks False

# 下载chatglm3
mkdir -p THUDM/chatglm3-6b
huggingface-cli download --resume-download THUDM/chatglm3-6b --local-dir THUDM/chatglm3-6b --local-dir-use-symlinks False

# 下载chatglm4
mkdir -p THUDM/glm-4-9b-chat
huggingface-cli download --resume-download THUDM/glm-4-9b-chat --local-dir THUDM/glm-4-9b-chat --local-dir-use-symlinks False

# 下载qwen3
mkdir -p Qwen/Qwen3-8B
huggingface-cli download --resume-download Qwen/Qwen3-8B-Instruct --local-dir Qwen/Qwen3-8B --local-dir-use-symlinks False

# 下载 lmsys/vicuna-13b-v1.5
mkdir -p lmsys/vicuna-7b-v1.5
huggingface-cli download --resume-download lmsys/vicuna-7b-v1.5 --local-dir lmsys/vicuna-7b-v1.5 --local-dir-use-symlinks False

# 下载01-ai/Yi-6B
mkdir -p 01-ai/Yi-6B
huggingface-cli download --resume-download 01-ai/Yi-6B --local-dir 01-ai/Yi-6B --local-dir-use-symlinks False

# 下载智谱ai BAAI/AquilaChat-7B
mkdir -p BAAI/AquilaChat-7B
huggingface-cli download --resume-download BAAI/AquilaChat-7B --local-dir BAAI/AquilaChat-7B --local-dir-use-symlinks False

# 下载google/gemma-7b-it
mkdir -p google/gemma-7b
huggingface-cli download --resume-download google/gemma-7b-it --local-dir google/gemma-7b-it --local-dir-use-symlinks False

# 下载bigscience/bloomz
mkdir -p bigscience/bloomz
huggingface-cli download --resume-download bigscience/bloomz --local-dir bigscience/bloomz --local-dir-use-symlinks False

# 下载llama2
mkdir -p meta-llama/Llama-2-7b-hf
huggingface-cli download --resume-download meta-llama/Llama-2-7b-hf --local-dir meta-llama/Llama-2-7b-hf --local-dir-use-symlinks False

# 下载llama3
mkdir -p meta-llama/Meta-Llama-3-8B
huggingface-cli download --resume-download meta-llama/Meta-Llama-3-8B --local-dir meta-llama/Meta-Llama-3-8B --local-dir-use-symlinks False

