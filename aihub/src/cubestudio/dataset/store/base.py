
import os
import sys
import time
import logging


class Store_client():
    def __init__(self,**kwargs):
        pass

    # 获取数据列表
    def file_list(self):
        return []

    # 删除文件
    def batch_delete(self,object_list):
        pass

    def get_download_url(self,remote_file_path):
        return ''
    # 下载图片
    def downloadfiles(self,remote_paths,lical_dir):
        pass

    # 上传单个文件
    def uploadfile(self,local_file_path,remote_file_path):
        pass

    # 上传图片，只上传文件夹下的文件
    def uploadfiles(self,local_dir,remote_dir):
        pass

    # 显示在bucket上的所有文件路径.prefix 表示目录
    def showFiles(self,prefix=''):
        pass


