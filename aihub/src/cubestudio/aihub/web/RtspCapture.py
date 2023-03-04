import os
import cv2
import time
import threading
import queue


class RtspCapture(object):
    read_thread = None  # background thread that reads frames from camera
    get_thread = None  # 从imglist中获取帧的进程
    pop_frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera
    imgList = None
    url = "rtsp://admin:admin888@192.168.1.64:554/h264/ch1/sub/av_stream"
    top = 100
    lock = None

    def __init__(self,url):
        self.top = 100
        self.url=url

    def start(self):
        # self.imgList = Manager().list()
        self.imgList = queue.Queue(self.top)
        self.lock = threading.Lock()
        if self.read_thread is None:
            # start background frame thread
            self.read_thread = threading.Thread(target=self.read)
            self.read_thread.start()

        if self.get_thread is None:
            self.last_access = time.time()
            self.get_thread = threading.Thread(target=self.get_frame)
            self.get_thread.start()

    def set_url(self, src):
        self.url = src

    # 向共享缓冲栈中写入数据:
    def read(self):
        print('Process to write: %s' % os.getpid())
        cap = cv2.VideoCapture(self.url)
        while True:
            _, img = cap.read()
            if _:
                if self.imgList.full():
                    self.imgList.get()
                self.imgList.put(img)

    # 在缓冲栈中读取数据:
    def get_frame(self):
        print('Process to get: %s' % os.getpid())
        while True:
            if not self.imgList.empty():
                self.lock.acquire()
                value = self.imgList.get(False)  # 非阻塞方法
                self.pop_frame = value
                self.lock.release()

    def cap_frame(self):
        if self.pop_frame is None:
            print('frame is None')
        else:
            # print('set frame')
            self.lock.acquire()
            jpg = cv2.imencode('.jpg', self.pop_frame)[1].tobytes()
            self.lock.release()
            return jpg