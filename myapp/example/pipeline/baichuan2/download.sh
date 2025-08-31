pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT="https://hf-mirror.com"
mkdir -p baichuan-inc/Baichuan2-13B-Chat
huggingface-cli download --resume-download baichuan-inc/Baichuan2-13B-Chat --local-dir baichuan-inc/Baichuan2-13B-Chat --local-dir-use-symlinks False

