
pip3 install pysnooper flask requests Pillow torch torchvision opencv-python
apt install -y libgl1-mesa-glx
wget https://docker-76009.sz.gfp.tencent-cloud.com/github/cube-studio/pipeline/coco_data_sample.zip && unzip -o coco_data_sample.zip &&  mv coco_data_sample/yolo / && rm -rf coco_data_sample*
pip3 install https://github.com/danielgatis/darknetpy/raw/master/dist/darknetpy-4.1-cp36-cp36m-linux_x86_64.whl

echo -e "classes= 80\nnames = /yolo/coco.names\neval=coco" > /yolo/coco.data