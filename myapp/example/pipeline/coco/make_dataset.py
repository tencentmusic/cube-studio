import os,io,sys,datetime,time,json,pandas

all_images = os.listdir('images/train2014')

all_class=[ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

all_data = []
for file in all_images:
    file_name = file.split('.')[0]
    file_path = f'images/train2014/{file}'
    label_path = f'labels/train2014/{file_name}.txt'
    if os.path.exists(label_path):
        labels = open(label_path).readlines()
        labels = [x.strip() for x in labels if x.strip()]
        labels = [x.split(' ') for x in labels]
        for label in labels:
            label.insert(0,all_class[int(label[0])])
            label = [file_name,file_path]+label
            all_data.append(label)


df = pandas.DataFrame(all_data, columns=['id', 'image', 'class_name','class_index','x','y','width','height'])
df.to_csv('data.csv',index=False)
feature={
    "id": {
        "_type": "Value",
        "dtype": "string"
    },
    "image": {
        "_type": "Value",
        "dtype": "Image"
    },
    "class_name": {
        "_type": "Value",
        "dtype": "string"
    },
    "class_index": {
        "_type": "Value",
        "dtype": "int"
    },
    "x": {
        "_type": "Value",
        "dtype": "float"
    },
    "y": {
        "_type": "Value",
        "dtype": "float"
    },
    "width": {
        "_type": "Value",
        "dtype": "float"
    },
    "height": {
        "_type": "Value",
        "dtype": "float"
    }
}
json.dump(feature,open('data.json',mode='w'),indent=4)


