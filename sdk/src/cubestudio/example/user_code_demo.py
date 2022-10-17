# ===================用户代码=====================

import tensorflow as tf
import os
import numpy
from predict_model import Offline_Predict

class My_Offline_Predict(Offline_Predict):

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        gpus = tf.config.list_physical_devices('GPU')
        # print(gpus)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        self.model = tf.saved_model.load('/mnt/pengluan/ray/ori/')

    # 定义所有要处理的数据源，返回字符串列表
    def datasource(self):
        all_lines = open('/mnt/pengluan/ray/aa.txt', mode='r').readlines()
        return all_lines

    # 定义一条数据的处理逻辑
    def predict(self,value):
        feats_npy = numpy.load(value)
        result = self.model(feats_npy)
        print(result)
        return result

My_Offline_Predict().run()


