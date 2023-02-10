
import os,sys,json,time,random,io,base64

import base64
import os
import datetime
import logging


# 本地数据集，需要完成类似numpy相似的功能。
class LocalDataset():
    dataset = None
    data = None
    data_type = ''  # audio,image,text,video

    cache_files = ''
    num_columns=0
    num_rows=0
    shape=[]

    def __init__(self):
        pass
        # todo 支持本地csv，json，txt等数据类型。(媒体文件，音频，视频，图片以txt的形式存在)
        # data = load_local_file()
        # todo 媒体文件原本txt列，转化为对应的类型 列

    # todo 从csv中加载数据集
    def load_from_csv(self):
        pass

    # todo 从json中加载数据集
    def load_from_json(self):
        pass

    # todo 从txt中加载数据集
    def load_from_txt(self):
        pass


    # todo numpy类型列操作sort shuffle select filter Shard(抽样) train_test_split
    # def sort(self):
    #     pass

    # todo 数据集的索引(按行按列)读取数据 dataset[0]  dataset["text"]
    # todo 数据集的分片  dataset[:3]

    # todo 列增删操作 rename_column remove_columns Cast类型转换 cast_column Flatten map

    # todo 支持数据 列
    def map(self):
        pass

    # todo 针对不同训练框架，转化为tf pytorch的数据集
    def to_tf_dataset(self):
        pass
    def to_torch_dataset(self):
        pass
    def to_numpy_dataset(self):
        pass

    # todo 数据分布的统计度量
    def metric(self):
        {
            "num_rows":0,
            "num_columns":0
        }

        pass

    # todo 数据分布统计可视化 pyechart,需要先做metric度量
    def echart(self):
        pass

    # todo 生成样例，根据自定义的
    def examples(self,num=100):
        # todo 根据用户自定义返回样例
        # if self.dataset.DatasetBuilder:
        #     return self.dataset.DatasetBuilder(self.dataset)
        pass
        # todo 按格式返回样例


    # todo 获取基础元素的info，由派生类实现
    def item_info(self):
        pass

    # todo 追加数据
    def add_item(self):
        pass

    # todo 追加列
    def add_column(self):
        pass

    # todo 合并标注。，由派生类实现
    def merge_label(self):
        pass


    def to_pandas(self):
        pass
    def to_json(self):
        pass
    def to_dict(self):
        pass
    def to_parquet(self):
        pass

