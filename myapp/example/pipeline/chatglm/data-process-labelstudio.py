import json
import os,time,datetime
import shutil

import requests
# nlp 问答自动化标注 对接chatglm训练

predictions_file = '/mnt/admin/pipeline/example/chatglm/predictions.json'
label_dir = '/mnt/admin/pipeline/example/chatglm/label-result/'
train_dir = '/mnt/admin/pipeline/example/chatglm/train.txt'
all_annotations={
}
instruction="你现在是一个问答模型，用于回答关于GitHub项目cube-studio的信息。文本："

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
# chatglm目标格式

# 处理标注结果
all_train_image=[]
file = open(train_dir,mode='w')
for task_id in all_annotations:
    annotation = all_annotations[task_id]
    data = annotation['data']
    annotation = annotation['annotations']
    question = data['question']
    result = annotation['result']
    answer = ''
    if result:
        answer = result[0]['value']['text'][0]
    if not answer:
        continue
    train_one = {
        "instruction": instruction,
        "input": question,
        "output": answer
    }
    file.write(json.dumps(train_one,ensure_ascii=False))
    file.write('\n')
file.close()







