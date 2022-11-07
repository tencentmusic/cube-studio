pip3 install pysnooper flask requests Pillow PaddlePaddle paddleseg
git clone https://github.com/PaddlePaddle/PaddleSeg.git
pip install -r PaddleSeg/requirements.txt
python PaddleSeg/contrib/PP-HumanSeg/src/download_inference_models.py
python PaddleSeg/contrib/PP-HumanSeg/src/download_data.py

