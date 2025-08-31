export HF_ENDPOINT=https://hf-mirror.com
apt update -y
apt install -y aria2 git-lfs wget
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh
./hfd.sh Qwen/Qwen3-8B --tool aria2c -x 8

sed -i "s|bfloat16|float16|g"  Qwen3-8B/config.json

#./hfd.sh Qwen/QwQ-32B --tool aria2c -x 8
#sed -i "s|bfloat16|float16|g"  QwQ-32B/config.json
