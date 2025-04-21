import json
import os,time,datetime
import shutil

import requests

all_classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

predictions_file = '/mnt/admin/pipeline/example/yolo/predictions.json'
label_dir = '/mnt/admin/pipeline/example/yolo/coco-result/'
train_dir = '/mnt/admin/pipeline/example/yolo/dataset/'
all_annotations={
}

# 遍历所有之前的历史标注
if os.path.exists(predictions_file):
    predictions = json.load(open(predictions_file))
    for prediction in predictions:
        predictions_reuslt = prediction.get('annotations',prediction.get("predictions",[]))
        if predictions_reuslt:
            predictions_reuslt = predictions_reuslt[-1]  # 最后一个是最新标注的
            all_annotations[prediction['id']]={
                "annotations":predictions_reuslt,
                "data":prediction['data']
            }


# 遍历更新的标注结果
if os.path.exists(label_dir) and os.path.isdir(label_dir):
    label_files = os.listdir(label_dir)
    for file in label_files:
        if not file.isdigit():
            continue

        file_path = os.path.join(label_dir,file)
        label = json.load(open(file_path))
        task_id = label['task']['id']
        data = label['task']['data']
        if task_id not in all_annotations:
            all_annotations[task_id]={
                "annotations": label,
                "data":data
            }
        else:
            # 如果是更新到标注结果，就替换
            if label['updated_at']>all_annotations[task_id]['annotations']['updated_at']:
                all_annotations[task_id] = {
                    "annotations": label,
                    "data": data
                }


# 将标注结果转化为目标格式
# yolo目标格式
os.makedirs(os.path.join(train_dir,'images/train'),exist_ok=True)
os.makedirs(os.path.join(train_dir,'labels/train'),exist_ok=True)


# 处理标注结果
all_train_image=[]
for task_id in all_annotations:
    annotation = all_annotations[task_id]
    data = annotation['data']
    annotation = annotation['annotations']
    image = data['image']
    name = ''
    # 先保存图片
    if 'https://' in image or 'http://' in image:
        response = requests.get(image)
        name = image[image.rindex("/") + 1:]
        image_path = os.path.join(train_dir,'images/train',name)

        if os.path.exists(image_path):
            os.remove(image_path)

        # 确保请求成功
        if response.status_code == 200:
            # 将视频内容写入本地文件
            with open(image_path, "wb") as file:
                file.write(response.content)
                all_train_image.append(image_path)
                print(f"文件已成功保存到: {image_path}")
        else:

            print(f"请求失败，状态码: {response.status_code}")
    else:
        if os.path.exists(image):
            name = os.path.basename(image)
            image_path = os.path.join(train_dir, 'images/train', name)
            if os.path.exists(image_path):
                os.remove(image_path)
            shutil.copy(image,image_path)
            all_train_image.append(image_path)
        else:
            print(f'文件不存在或无法访问，{image}')

    # 再保存label
    if not name:
        continue
    name = name[:name.index('.')]
    # if 'result' not in annotation:
    #     print(annotation)
    #     break
    labels = annotation['result']
    path = os.path.join(train_dir, 'labels/train', name+".txt")
    if os.path.exists(path):
        os.remove(path)
    save_labels=[]
    for label in labels:
        if label['type']=='rectanglelabels':
            class_type = label['value']['rectanglelabels']
            class_type = class_type[0]
            class_type_index = all_classes.index(class_type)
            x1,y1,width,height = label['value']['x']/100,label['value']['y']/100,label['value']['width']/100,label['value']['height']/100
            x2,y2=x1+width,y1+height
            x_center = (x1+x2)/2
            y_center = (y1+y2)/2
            save_label = [class_type_index,round(x_center,6),round(y_center,6),round(width,6),round(height,6)]
            save_label = [str(x) for x in save_label]
            save_label = ' '.join(save_label)
            save_labels.append(save_label)
    file = open(path,mode='w')
    print('write:',path)

    file.write('\n'.join(save_labels))
    file.close()

train_path = os.path.join(train_dir,'train.txt')
if os.path.exists(train_path):
    os.remove(train_path)
file = open(train_path,mode='w')
for image in all_train_image:
    file.write(image+'\n')
file.close()






