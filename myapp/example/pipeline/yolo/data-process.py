import json
import os
base_dir = '/mnt/admin/coco'
label_files = os.listdir(base_dir)

for file in label_files:
    file_path = os.path.join(base_dir,file)
    label = json.load(open(file_path))



