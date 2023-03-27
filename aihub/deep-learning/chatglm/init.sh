
# 安装基础环境
pip3 install opencv-python install "opencv-python-headless<4.3"

# 下载模型
cd /app/ && python download_model.py
mv /root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/*/* /models/
