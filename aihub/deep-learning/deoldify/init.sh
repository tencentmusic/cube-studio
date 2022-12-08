
git clone https://github.com/jantic/DeOldify.git
pip3 install -r DeOldify/requirements.txt
mkdir /DeOldify/models
wget https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth -O /DeOldify/models/ColorizeArtistic_gen.pth
# 下载预训练模型
cd /app
python -c "from app import model;model.load_model()"