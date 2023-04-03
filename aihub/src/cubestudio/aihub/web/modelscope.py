import os,pysnooper

import numpy,cv2,time,random

@pysnooper.snoop()
def draw_image(input_path, result):
    def get_color(idx):
        idx = (idx + 1) * 3
        color = ((10 * idx) % 255, (20 * idx) % 255, (30 * idx) % 255)
        return color

    if 'http' in input_path:
        import requests
        os.makedirs('result', exist_ok=True)
        save_path = 'result/' + str(random.randint(1, 10000)) + ".jpg"
        open(save_path, 'wb').write(requests.get(input_path).content)
        img = cv2.imread(save_path)
    else:
        img = cv2.imread(input_path)

    rows = []
    if result.get('scores', []):
        rows = result['scores']
    elif result.get('boxes',[]):
        rows=result['boxes']
    elif result.get('keypoints', []):
        rows = result['keypoints']
    elif result.get('labels', []):
        rows = result['labels']

    for idx, row in enumerate(rows):
        score = round(result['score'][idx] if result.get('score', '') else 0,2)
        box = result['boxes'][idx] if result.get('boxes', []) else []
        if type(box) is numpy.ndarray:
            box = box.tolist()
        keypoint = result['keypoints'][idx] if result.get('keypoints', []) else []
        if type(keypoint) is numpy.ndarray:
            keypoint = keypoint.tolist()
        label = result['labels'][idx] if result.get('labels', '') else ''
        color = get_color(idx)
        text_size = 0.001 * (img.shape[0] + img.shape[1]) / 2 + 0.3
        line_width = int(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        if box:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, line_width)
        if box and (label or score):
            text = label + str(score)
            cv2.putText(img, text, (int(box[0]), int(box[1]) + 10), cv2.FONT_HERSHEY_PLAIN, text_size, color,
                        line_width)
        if keypoint:
            point = keypoint[0]
            if type(point)==list: # 如果x，y放在一起
                x = [point[0] for point in keypoint]
                y = [point[1] for point in keypoint]
                radius = int(max(1, (max(x) - min(x)) // 50, (max(y) - min(y)) // 50))
                for index, dot in enumerate(x):
                    cv2.circle(img, (int(x[index]), int(y[index])), radius, (0, 0, 255), -1)

            # 如果x，y并列排放
            else:
                x = [keypoint[index] for index in range(len(keypoint) // 2)]
                y = [keypoint[index + 1] for index in range(len(keypoint) // 2)]
                radius = int(max(1, (max(x) - min(x)) // 50, (max(y) - min(y)) // 50))
                for index, dot in enumerate(x):
                    cv2.circle(img, (int(x[index]), int(y[index])), radius, (0, 0, 255), -1)

    return img