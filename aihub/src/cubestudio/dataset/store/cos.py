import datetime
import os
import sys

import requests
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import logging
import time
import logging

import pysnooper
from .base import Store_client

# @pysnooper.snoop()
class COS_client(Store_client):

    def __init__(self,appid,secret_id,secret_key,bucket_name,endpoint=None,region=None,root='/dataset',token=None,timeout=30,internal=False,**kwargs):
        super(COS_client, self).__init__()
        self.appid=appid
        self.secret_id=secret_id
        self.secret_key=secret_key
        self.region=region
        self.token=token
        self.bucket_name=bucket_name
        self.root=root
        self.config = CosConfig(Appid=appid,Region=region,Endpoint=endpoint, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme="https",Timeout=timeout)
        self.client = CosS3Client(self.config)
        self.download_host = kwargs.get('download_host','')

    def upload_percentage(self,consumed_bytes, total_bytes):
        """进度条回调函数，计算当前上传的百分比

        :param consumed_bytes: 已经上传的数据量
        :param total_bytes: 总数据量
        """
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            print('\r{0}% '.format(rate))
            sys.stdout.flush()

    # 上传单个文件
    # @pysnooper.snoop()
    def uploadfile(self,local_file_path,remote_file_path):
        remote_file_path = remote_file_path.lstrip()
        if 'http://' in local_file_path or 'https://' in local_file_path:
            local_file_path_name = local_file_path[local_file_path.rfind('/')+1:]
            if not local_file_path_name:
                local_file_path_name = int(datetime.datetime.now().timestamp()*1000)

            res = requests.get(local_file_path, stream=True)
            f = open(local_file_path_name, "wb")
            for chunk in res.iter_content(chunk_size=5120):
                if chunk:
                    f.write(chunk)

            if res.status_code==200:
                local_file_path = local_file_path_name
        print(local_file_path,remote_file_path)
        response = self.client.upload_file(
            Bucket=self.bucket_name,
            LocalFilePath=local_file_path,
            Key=remote_file_path,
            PartSize=5,
            MAXThread=10,
            EnableMD5=False,
            progress_callback=self.upload_percentage,
        )
        print(response)
        print(response['ETag'])

    # @pysnooper.snoop()
    def get_download_url(self,remote_file_path):
        marker = ""
        file_list = []
        while True:
            response = self.client.list_objects(
                Bucket=self.bucket_name,
                Prefix=remote_file_path.lstrip('/'),
                Marker=marker
            )
            print(response)
            file_list+=[file['Key'] for file in response['Contents']]
            if response['IsTruncated'] == 'false':
                break
            marker = response['NextMarker']

        file_list = [self.download_host.strip('/')+"/"+url for url in file_list]
        return file_list



