#coding=utf-8
#绘制面部轮廓
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import base64,numpy,os
import pysnooper,datetime,time
from io import BytesIO
from flask import Flask, render_template, request, Response, jsonify
from flask_cors import CORS
import json
import cv2,dlib
from keras.models import load_model
import datetime
import numpy as np
import sys
import os
import requests,json,datetime,time,os,sys
import shutil,pysnooper
import logging
import base64

app = Flask(__name__)
CORS(app, supports_credentials=True)
# base_dir = os.path.abspath(__file__)
# # print(base_dir)
base_dir = os.path.split(os.path.realpath(__file__))[0]
print(base_dir)



# @pysnooper.snoop()
def draw(image_np):

    image = image_np #  face_recognition.load_image_file("static/img/ag.png")

    #查找图像中所有面部的所有面部特征
    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        facial_features = [
            'chin',  # 下巴
            'left_eyebrow',  # 左眉毛
            'right_eyebrow',  # 右眉毛
            'nose_bridge',  # 鼻樑
            'nose_tip',  # 鼻尖
            'left_eye',  # 左眼
            'right_eye',  # 右眼
            'top_lip',  # 上嘴唇
            'bottom_lip'  # 下嘴唇
        ]

        '''
        PIL会失色,opencv不会
        pil_image = Image.fromarray(image,mode='RGB')
        d = ImageDraw.Draw(pil_image,mode='RGB')
        for facial_feature in facial_features:
            d.line(face_landmarks[facial_feature], fill=(255, 255, 255), width=2)

        # pil_image.save('face/%s.jpg'%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        output_buffer = BytesIO()
        pil_image.save(output_buffer, format='JPEG')
        byte_data = output_buffer.getvalue()
        base64_byte = base64.b64encode(byte_data)
        base64_str = 'data:image/jpg;base64,'+str(base64_byte,encoding='utf-8')
        '''
        # pil_image = Image.fromarray(image, mode='RGB')
        # d = ImageDraw.Draw(image,mode='RGB')

        print(face_landmarks[facial_features[0]])
        for facial_feature in facial_features:
            cv2.polylines(image,np.array([face_landmarks[facial_feature]]),False, color=(255, 255, 255))
        # img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        base64_byte = cv2.imencode('.jpg', image)[1].tostring()
        base64_byte = base64.b64encode(base64_byte)
        base64_str = 'data:image/jpg;base64,'+str(base64_byte,encoding='utf-8')

        return base64_str


@app.route('/')
def index():
    period = int(os.getenv('PERIOD','100'))
    return render_template('camera.html',period=period)

@app.route('/hello')
def hello():
    return Response('hello_world')



@app.route('/receiveImage/', methods=["POST"])
def receive_image():

    str_image = request.data.decode('utf-8')
    img = base64.b64decode(str_image)
    img_np = numpy.fromstring(img, dtype='uint8')
    new_img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    base64_str = draw(new_img_np)
    # cv2.imwrite('./images/rev_image.jpg', new_img_np)
    # print('data:{}'.format('success'))


    # return Response('data:image/jpg;base64,'+str_image)
    return Response(base64_str)


#人脸分类器
detector = dlib.get_frontal_face_detector()
# 获取人脸检测器
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

@app.route('/autodetectFace/',methods=["POST"])
def autodetectFace():
    str_image = request.data.decode('utf-8')
    img = base64.b64decode(str_image)
    img_np = numpy.fromstring(img, dtype='uint8')
    new_img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)


    gray = cv2.cvtColor(new_img_np, cv2.COLOR_BGR2GRAY)


    dets = detector(gray, 1)
    for face in dets:
        # 在图片中标注人脸，并显示
        # left = face.left()
        # top = face.top()
        # right = face.right()
        # bottom = face.bottom()
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        # cv2.imshow("image", img)

        shape = predictor(new_img_np, face)  # 寻找人脸的68个标定点
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(new_img_np, pt_pos, 1, (0, 255, 0), 2)
        #cv2.imshow("image", img)

        '''
        PIL会失色
        
        pil_image = Image.fromarray(new_img_np, mode='RGB')
        #pil_image.save("ag11111111111111111.png")

        output_buffer = BytesIO()
        pil_image.save(output_buffer, format='JPEG')
        byte_data = output_buffer.getvalue()
        base64_byte = base64.b64encode(byte_data)
        base64_str = 'data:image/jpg;base64,'+str(base64_byte,encoding='utf-8')
        '''

        base64_byte = cv2.imencode('.jpg', new_img_np)[1].tostring()
        base64_byte = base64.b64encode(base64_byte)
        base64_str = 'data:image/jpg;base64,'+str(base64_byte,encoding='utf-8')

        return Response(base64_str)

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "static/font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


emotion_classifier = load_model(os.path.join(base_dir,'simple_CNN.530-0.65.hdf5'))

face_classifier = cv2.CascadeClassifier(os.path.join(base_dir,"haarcascade_frontalface_default.xml"))
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

graph = tf.get_default_graph()

emotion_labels = {
    0: '生气',
    1: '厌恶',
    2: '恐惧',
    3: '开心',
    4: '难过',
    5: '惊喜',
    6: '平静'
}

# @pysnooper.snoop()
def get_em(img):

    # img = cv2.imread("/app/static/img/emotion.png")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
    color = (255, 0, 0)
    with graph.as_default():
        for (x, y, w, h) in faces:
            gray_face = gray[(y):(y + h), (x):(x + w)]
            gray_face = cv2.resize(gray_face, (48, 48))
            gray_face = gray_face / 255.0
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion = emotion_labels[emotion_label_arg]
            cv2.rectangle(img, (x + 10, y + 10), (x + h - 10, y + w - 10),
                          (255, 255, 255), 2)
            img = cv2ImgAddText(img, emotion, x + h * 0.3, y, color, 20)

            '''
            PIL会失色,opencv不会
            
            pil_image = Image.fromarray(img, mode='RGB')
            # pil_image.save("ag11111111111111111.png")

            output_buffer = BytesIO()
            pil_image.save(output_buffer, format='JPEG')
            byte_data = output_buffer.getvalue()
            base64_byte = base64.b64encode(byte_data)
            base64_str = 'data:image/jpg;base64,'+str(base64_byte,encoding='utf-8')
            '''

            base64_byte = cv2.imencode('.jpg', img)[1].tostring()
            base64_byte = base64.b64encode(base64_byte)
            base64_str = 'data:image/jpg;base64,' + str(base64_byte, encoding='utf-8')
            return base64_str

@app.route('/emotionRecognition/',methods=["POST"])
def emotionRecognition():

    str_image = request.data.decode('utf-8')
    img = base64.b64decode(str_image)
    img_np = numpy.fromstring(img, dtype='uint8')
    new_img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    base64_str=get_em(new_img_np)

    return Response(base64_str)


@app.route('/faceswap/',methods=["POST"])
def faceswap():
    # modelPath = "../app/shape_predictor_68_face_landmarks.dat"
    SCALE_FACTOR = 1
    FEATHER_AMOUNT = 11

    FACE_POINTS = list(range(17, 68))
    MOUTH_POINTS = list(range(48, 61))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))
    JAW_POINTS = list(range(0, 17))

    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

    OVERLAY_POINTS = [
        LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
        NOSE_POINTS + MOUTH_POINTS,
    ]

    COLOUR_CORRECT_BLUR_FRAC = 0.6

    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(modelPath)

    class TooManyFaces(Exception):
        pass

    class NoFaces(Exception):
        pass

    def get_landmarks(im):
        rects = detector(im, 1)

        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces

        return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

    def annotate_landmarks(im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(
                im,
                str(idx),
                pos,
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                fontScale=0.4,
                color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im

    def draw_convex_hull(im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def get_face_mask(im, landmarks):
        im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

        for group in OVERLAY_POINTS:
            draw_convex_hull(im, landmarks[group], color=1)

        im = numpy.array([im, im, im]).transpose((1, 2, 0))

        im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

        return im

    def transformation_from_points(points1, points2):
        points1 = points1.astype(numpy.float64)
        points2 = points2.astype(numpy.float64)
        c1 = numpy.mean(points1, axis=0)
        c2 = numpy.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = numpy.std(points1)
        s2 = numpy.std(points2)
        points1 /= s1
        points2 /= s2
        U, S, Vt = numpy.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        return numpy.vstack([
            numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
            numpy.matrix([0., 0., 1.])
        ])

    def read_im_and_landmarks(fname=None):
        if fname:
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
        else:
            str_image = request.data.decode('utf-8')
            img = base64.b64decode(str_image)
            img_np = numpy.fromstring(img, dtype='uint8')
            im = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        im = cv2.resize(im,
                        (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
        s = get_landmarks(im)

        return im, s

    def warp_im(im, M, dshape):
        output_im = numpy.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(
            im,
            M[:2], (dshape[1], dshape[0]),
            dst=output_im,
            borderMode=cv2.BORDER_TRANSPARENT,
            flags=cv2.WARP_INVERSE_MAP)
        return output_im

    def correct_colours(im1, im2, landmarks1):
        blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
            numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
            numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                im2_blur.astype(numpy.float64))
    # '''
    im1, landmarks1 = read_im_and_landmarks(os.path.join(base_dir,"static/img/ag-2.png"))
    # im2, landmarks2 = read_im_and_landmarks("static/img/ag.png")
    im2, landmarks2 = read_im_and_landmarks()
    '''
    # im1, landmarks1 = read_im_and_landmarks("static/img/ag-2.png")
    im2, landmarks2 = read_im_and_landmarks("static/img/ag-2.png")
    im1, landmarks1 = read_im_and_landmarks()
    '''

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])

    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = numpy.max(
        [get_face_mask(im1, landmarks1), warped_mask], axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

    # cv2.imwrite("static/img/faceswap1.png", output_im)

    '''
    这里是用PIL输出base64编码,会失去图片原来的颜色
    
    pil_image = Image.fromarray(np.uint8(output_im),mode='RGB')
    pil_image.save("ag111111111111111111112.png")

    output_buffer = BytesIO()
    pil_image.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_byte = base64.b64encode(byte_data)
    base64_str = 'data:image/jpg;base64,' + str(base64_byte, encoding='utf-8')
    '''

    base64_byte = cv2.imencode('.jpg', output_im)[1].tostring()
    base64_byte = base64.b64encode(base64_byte)
    base64_str = 'data:image/jpg;base64,' + str(base64_byte, encoding='utf-8')
    # output_buffer = BytesIO()
    # cv_img.save(output_buffer, format='JPEG')
    # byte_data = output_buffer.getvalue()
    # base64_byte = base64.b64encode(cv_img)
    # base64_str = 'data:image/jpg;base64,' + str(base64_byte, encoding='utf-8')
    return base64_str


#
# total_image_name = []
# total_face_encoding = []
#
# def load_local_face():
#     path = "static/img/face_recognition"  # 模型数据图片目录
#
#     for fn in os.listdir(path):  # fn 表示的是文件名q
#         print(path + "/" + fn)
#         img_path = path + "/" + fn
#         if '.png' not in img_path:
#             os.remove(img_path)
#             continue
#         face_img = face_recognition.load_image_file(img_path)
#
#         faces = face_recognition.face_encodings(face_img)
#         if faces and len(faces)>0:
#             total_face_encoding.append(faces[0])
#             fn = fn[:(len(fn) - 4)]  # 截取图片名（这里应该把images文件中的图片名命名为为人物名）
#             total_image_name.append(fn)  # 图片名字列表
#         else:
#             os.remove(img_path)
#
# load_local_face()
#
# @app.route('/facesearch/',methods=["POST"])
# def facesearch():
#     str_image = request.data.decode('utf-8')
#     img = base64.b64decode(str_image)
#     img_np = numpy.fromstring(img, dtype='uint8')
#     frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#
#     # 发现在视频帧所有的脸和face_enqcodings
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)
#
#     # 在这个视频帧中循环遍历每个人脸
#     for (top, right, bottom, left), face_encoding in zip(
#             face_locations, face_encodings):
#         # 看看面部是否与已知人脸相匹配。
#         name = "Unknown"
#         result = face_recognition.face_distance(total_face_encoding,face_encoding).tolist()
#         # print(result)
#         xiangsidu = min(result)
#         index = result.index(xiangsidu)
#         if xiangsidu<0.4:
#             name = total_image_name[index]
#
#         # for i, v in enumerate(total_face_encoding):
#         #     match = face_recognition.compare_faces(
#         #         [v], face_encoding, tolerance=0.4)
#         #     # print(i,v)
#         #     # print(match)
#         #     if match[0]:
#         #         name = total_image_name[i]
#                 # break
#         # 画出一个框，框住脸
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         # 画出一个带名字的标签，放在框下
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255),
#                       cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0,
#                     (255, 255, 255), 1)
#     # 显示结果图像
#     base64_byte = cv2.imencode('.jpg', frame)[1].tostring()
#     base64_byte = base64.b64encode(base64_byte)
#     base64_str = 'data:image/jpg;base64,' + str(base64_byte, encoding='utf-8')
#     return Response(base64_str)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
# get_em()



