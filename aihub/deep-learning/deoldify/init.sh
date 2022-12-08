
git clone https://github.com/jantic/DeOldify.git
pip3 install -r DeOldify/requirements.txt
mkdir /DeOldify/models
wget https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth -O /DeOldify/models/ColorizeArtistic_gen.pth
python /app/load.py