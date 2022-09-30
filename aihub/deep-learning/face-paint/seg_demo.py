import argparse
import base64
import datetime
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm


from paddleseg.utils import get_sys_env, logger, get_image_list

from infer import Predictor

import os
import dlib
import collections
from typing import Union, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from PIL import Image


def get_bg_img(bg_img_path, img_shape):
    if bg_img_path is None:
        bg = 255 * np.ones(img_shape)
    elif not os.path.exists(bg_img_path):
        raise Exception('The --bg_img_path is not existed: {}'.format(
            bg_img_path))
    else:
        bg = cv2.imread(bg_img_path)
    return bg


def makedirs(save_dir):
    dirname = save_dir if os.path.isdir(save_dir) else \
        os.path.dirname(save_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def seg_image(args):
    assert os.path.exists(args['img_path']), \
        "The --img_path is not existed: {}.".format(args['img_path'])

    logger.info("Input: image")
    logger.info("Create predictor...")
    predictor = Predictor(args)

    logger.info("Start predicting...")
    img = cv2.imread(args['re_save_path'])
    bg_img = get_bg_img(args['bg_img_path'], img.shape)
    out_img = predictor.run(img, bg_img)
    # print(type(out_img))
    cv2.imwrite(args['save_dir'], out_img)
    file = open(args['save_dir'], 'rb')
    base64_str = base64.b64encode(file.read()).decode('utf-8')
    print(len(base64_str))
    return base64_str
    # img_ = Image.open(out_img)
    # print(img_)



def get_dlib_face_detector(predictor_path: str = "shape_predictor_68_face_landmarks.dat"):
    if not os.path.isfile(predictor_path):
        model_file = "shape_predictor_68_face_landmarks.dat.bz2"
        os.system(f"wget http://dlib.net/files/{model_file}")
        os.system(f"bzip2 -dk {model_file}")

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def detect_face_landmarks(img: Union[Image.Image, np.ndarray]):
        if isinstance(img, Image.Image):
            img = np.array(img)
        faces = []
        dets = detector(img)
        for d in dets:
            shape = shape_predictor(img, d)
            faces.append(np.array([[v.x, v.y] for v in shape.parts()]))
        return faces

    return detect_face_landmarks


def display_facial_landmarks(
        img: Image,
        landmarks: List[np.ndarray],
        fig_size=[15, 15]
):
    plot_style = dict(
        marker='o',
        markersize=4,
        linestyle='-',
        lw=2
    )
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {
        'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
        'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
        'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
        'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
        'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
        'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
        'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
        'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
        'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
    }

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis('off')

    for face in landmarks:
        for pred_type in pred_types.values():
            ax.plot(
                face[pred_type.slice, 0],
                face[pred_type.slice, 1],
                color=pred_type.color, **plot_style
            )
    plt.show()


import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage


def align_and_crop_face(
        img: Image.Image,
        landmarks: np.ndarray,
        expand: float = 1.0,
        output_size: int = 1024,
        transform_size: int = 4096,
        enable_padding: bool = True,
):
    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = landmarks
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= expand
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    return img


def start(
        config=r'/home/JLWL/PaddleSeg-release-2.6/contrib/PP-HumanSeg/src/inference_models/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax/deploy.yaml',
        img_path=r'/home/JLWL/PaddleSeg-release-2.6/contrib/PP-HumanSeg/src/data/images/1.jpg',
        bg_img_path=r'/home/JLWL/PaddleSeg-release-2.6/contrib/PP-HumanSeg/src/data/images/2.jpg',
        re_save_path=r'temp/1_.jpg',
        save_dir=r'temp/1.jpg',
        use_gpu=True,
        test_speed=False, use_optic_flow=False, use_post_process=False):
    args = {
        'config': config,
        'img_path': img_path,
        'bg_img_path': bg_img_path,
        're_save_path': re_save_path,
        'save_dir': save_dir,
        'use_gpu': use_gpu,
        'test_speed': test_speed,
        'use_optic_flow': use_optic_flow,
        'use_post_process': use_post_process
    }
    print(type(args))

    # 先动漫化后增加背景效果更佳

    # 加载网络或本地文件
    save_ = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    os.mkdir(save_)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", device=device).eval()
    face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device, side_by_side=True)
    img = Image.open(args['img_path']).convert("RGB")
    # img = Image.open("/content/sample.jpg").convert("RGB")

    face_detector = get_dlib_face_detector()
    landmarks = face_detector(img)
    out = ''
    for landmark in landmarks:
        face = align_and_crop_face(img, landmark, expand=1.3)
        p_face = face2paint(model=model, img=face, size=512)
        # display(p_face)
        # p_face.save('1.png') # 此输出为对比图片
        # 裁剪为需要的部分输出
        x_, y_ = p_face.size
        out = p_face.crop((int(x_ / 2), 0, x_, y_))
    img_ = out
    x, y = img_.size
    print(x, y)
    all_list = []
    for i in range(5):
        newIm = Image.new('RGB', (int(x * 1.5), int(y * 1.5)), 'white')
        newIm.paste(img_, (int(x * 0.5), int(y * 0.45)))
        # args['re_save_path'] = newIm
        newIm.save(args['re_save_path'])
        base64_ = seg_image(args)
        all_list.append({f'{i}': base64_})


if __name__ == "__main__":
    image_path = r'/home/JLWL/PaddleSeg-release-2.6/contrib/PP-HumanSeg/src/data/images/human.jpg'
    file_after = open(image_path, 'rb')
    base64_after_str = base64.b64encode(file_after.read()).decode('utf-8')
    print(len(base64_after_str))
    imgdata = base64.b64decode(base64_after_str)
    # 将图片保存为文件
    if os.path.exists('temp'):
        pass
    else:
        os.mkdir('temp')
    name_ = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    new_image_path = f'temp/{name_}.jpg'
    with open(new_image_path, 'wb') as f:
        f.write(imgdata)

    start(
        # config=r'E:\PaddleSeg-release-2.6\contrib\PP-HumanSeg\src\inference_models\portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax\deploy.yaml',
        img_path=new_image_path,
        # bg_img_path=r'E:\PaddleSeg-release-2.6\contrib\PP-HumanSeg\src\data\images\bg_1.jpg',
        # re_save_path=r'E:\PaddleSeg-release-2.6\contrib\PP-HumanSeg\src\data\images\_1.jpg',
        # save_dir=r'E:\PaddleSeg-release-2.6\contrib\PP-HumanSeg\src\data\images_result\1.jpg',
        # use_gpu=True,
        # test_speed=False, use_optic_flow=False, use_post_process=False)
        )
