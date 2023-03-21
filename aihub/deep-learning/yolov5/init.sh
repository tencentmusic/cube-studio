apt install zip
pip install matplotlib numpy opencv-python Pillow PyYAML requests scipy torch torchvision tqdm tensorboard pandas seaborn
git clone https://github.com/JLWLL/yolov5-6.1.git
mv yolov5-6.1 /yolov5
cd /app/ && python app.py download_model
