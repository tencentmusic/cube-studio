import logging
import os
import shutil
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
import requests
import pysnooper
import json
import importlib
import re,datetime,time,os,sys,io
import inspect
from cubestudio.request.model import Model
from cubestudio.utils.py_store import download_file,upload_file,decompress_file
from cubestudio.dataset.table import Table,InMemoryTable
from multiprocessing import Pool
from functools import partial
from pyarrow import csv


class Dataset(Model):
    path = '/dataset_modelview/api'

    # @property
    # def local_dir(self):
    #     return f"/data/k8s/kubeflow/dataset/{self.name}/{self.version}"
    local_dir = ''
    # @pysnooper.snoop()
    def download(self,partition='',des_dir=None):
        if des_dir:
            self.local_dir=des_dir
        else:
            self.local_dir=os.getcwd()
        print('准备下载数据到',self.local_dir)
        url = self.client.path+f"/download/{self.id}"
        if partition:
            url = url+"/"+partition
        donwload_urls = self.client.req(url).get("result",{}).get("download_urls",[])
        print(donwload_urls)
        os.makedirs(self.local_dir,exist_ok=True)
        pool = Pool(len(donwload_urls))  # 开辟包含指定数目线程的线程池
        pool.map(partial(download_file,des_dir=self.local_dir), donwload_urls)  # 当前worker，只处理分配给当前worker的任务
        pool.close()
        pool.join()

        # 下载info相关的文件，比如解析数据的代码，label代码，info.json

    # 上传数据部分
    # @pysnooper.snoop()
    def upload(self,file_path_list,partition=''):
        if type(file_path_list)!=list:
            file_path_list=[file_path_list]
        url = self.client.host.rstrip('/')+self.client.path + f"/upload/{self.id}"
        headers = {
            "Authorization":self.client.token
        }
        data = {
            "partition":partition
        }
        print('准备上传本地数据', file_path_list)
        pool = Pool(min(10,len(file_path_list)))  # 开辟包含指定数目线程的线程池
        pool.map(partial(upload_file,url=url,data=data,headers=headers), file_path_list)  # 当前worker，只处理分配给当前worker的任务
        pool.close()
        pool.join()
        pass

    # 文件加密
    def encrypt(self,file_path,save_path,key):
        from cryptography.fernet import Fernet
        f = Fernet(key)
        with open(file_path, 'rb') as original_file:
            original = original_file.read()

        encrypted = f.encrypt(original)
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path, 'wb') as encrypted_file:
            encrypted_file.write(encrypted)

    # 文件解压
    def compress(self,zip_path,compress_dir):
        shutil.make_archive(zip_path[:zip_path.index('.')],'zip',compress_dir)

    # 文件加密
    def decompress(self,zip_path,extract_dir=None):
        shutil.unpack_archive(zip_path, extract_dir=extract_dir, format=None)


    # 文件解密
    def decrypt(self,file_path,save_path,key):
        from cryptography.fernet import Fernet
        f = Fernet(key)

        with open(file_path, 'rb') as encrypted_file:
            encrypted = encrypted_file.read()

        decrypted = f.decrypt(encrypted)
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path, 'wb') as decrypted_file:
            decrypted_file.write(decrypted)

    # load成table结构
    # @pysnooper.snoop()
    def load(self,local_dir=None):
        if not local_dir:
            local_dir = self.local_dir
        self.local_dir=local_dir
        # # 解压压缩文件
        # files = os.listdir(local_dir)
        # files = [os.path.join(local_dir,filename) for filename in files if re.match('\.zip$',filename)  or re.match('\.tar\.gz$',filename)]
        # if files:
        #     pool = Pool(min(10, len(files)))  # 开辟包含指定数目线程的线程池
        #     pool.map(partial(decompress_file, des_dir=local_dir), files)  # 当前worker，只处理分配给当前worker的任务
        #     pool.close()
        #     pool.join()
        #     for path in files:
        #         os.remove(path)

        # 根据data.csv读取成table
        data_csv_path = os.path.join(local_dir,f'{self.name}.csv')
        if os.path.exists(data_csv_path):
            self.table = InMemoryTable(table=csv.read_csv(data_csv_path))
            self.table.local_dir = local_dir


        # 根据解析代码，解析成table
        extract_py_path = os.path.join(local_dir,f'{self.name}.py')
        if os.path.exists(extract_py_path):
            if not os.path.exists(os.path.join(local_dir,'__init__.py')):
                # 生成python包
                file = open(os.path.join(local_dir,'__init__.py'),mode='w')
                file.close()

            datase_builer = importlib.import_module(name=f"{self.name}.{self.name}")
            self.table = InMemoryTable(datase_builer.load(self.name))
            self.table.local_dir = local_dir

        # 将info信息，将数据列转化为特征
        info_path = os.path.join(local_dir, f'{self.name}.json')
        if os.path.exists(info_path):
            info = json.load(open(info_path))
            features = info.get("features",{})
            for feature_name in features:
                feature = features[feature_name]
                if feature_name in self.table.column_names:
                    _type = feature.get("_type",'Value')
                    del feature['_type']
                    import cubestudio.dataset.features as fea
                    self.table = self.table.cast_column(feature_name,getattr(fea,_type)(**feature))
                    self.table.local_dir = local_dir


    # todo 将数据集转存到云存储，cos，minio，oss等
    def save_cloud_storage(self):
        pass

    def __getitem__(self, key):
        """Can be used to index columns (by string names) or rows (by integer index or iterable of indices or bools)."""
        return self.table._getitem(
            key,
        )

    # table的函数
    # columns,num_columns，num_rows，column_names，shape,drop(col),rename_column(col),rename_columns(cols),filter,select,sort,shard,add_column,add_item

    # columns的函数
    # unique，cast_column,flatten