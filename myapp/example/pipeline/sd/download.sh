# 使用hfd.sh脚本下载
export HF_ENDPOINT=https://hf-mirror.com
apt update && apt install -y aria2 git-lfs wget axel
wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh

#./hfd.sh stabilityai/stable-diffusion-2-1 --tool aria2c -x 8
#./hfd.sh stabilityai/stable-diffusion-2 --tool aria2c -x 8
#./hfd.sh stabilityai/stable-diffusion-xl-base-1.0 --tool aria2c -x 8
#./hfd.sh runwayml/stable-diffusion-v1-5 --tool aria2c -x 8

#./hfd.sh stabilityai/stable-diffusion-2-depth --tool aria2c -x 8
#./hfd.sh stabilityai/stable-cascade-prior --tool aria2c -x 8
#./hfd.sh stabilityai/stable-cascade --tool aria2c -x 8


#mv stable-diffusion-2/768-v-ema.safetensors ./
#mv stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors ./
#mv stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors ./
#mv stable-diffusion-2-1/v2-1_768-ema-pruned.safetensors ./
#
#rm -rf stable-diffusion-2-1 stable-diffusion-v1-5 stable-diffusion-xl-base-1.0 stable-diffusion-2

aria2 -o v2-1_768-ema-pruned.safetensors https://hf-mirror.com/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors?download=true
aria2 -o 768-v-ema.safetensors https://hf-mirror.com/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.safetensors?download=true
aria2 -o sd_xl_base_1.0.safetensors https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true
aria2 -o v1-5-pruned-emaonly.safetensors https://hf-mirror.com/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors?download=true
