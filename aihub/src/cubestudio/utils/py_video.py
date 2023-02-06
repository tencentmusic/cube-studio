

# !/usr/bin/env python
# -*-coding:utf-8-*-
import time
import cv2
import os

def Video2Mp4(videoPath, outVideoPath):

    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input=videoPath, output=outVideoPath.replace(".mp4",'')))
    return True
    #
    # capture = cv2.VideoCapture(videoPath)
    # fps = capture.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # # fNUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # suc = capture.isOpened()  # 是否成功打开
    #
    # allFrame = []
    # while suc:
    #     suc, frame = capture.read()
    #     if suc:
    #         allFrame.append(frame)
    # capture.release()
    #
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # videoWriter = cv2.VideoWriter(outVideoPath, fourcc, fps, size)
    # for aFrame in allFrame:
    #     videoWriter.write(aFrame)
    # videoWriter.release()


import glob

def TS2Mp4(ts_path,mp4_path):
    with open(mp4_path, 'wb') as fw:
        files = glob.glob('{}/*.ts'.format(ts_path))
        files.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))
        for file in files:
            with open(file, 'rb') as fr:
                fw.write(fr.read())





if __name__ == '__main__':
    inputVideoPath = "mv_1639632186.3482149.avi"  # 读取视频路径
    outVideoPath = f"out_{int(time.time())}.mp4"
    Video2Mp4(inputVideoPath, outVideoPath)


