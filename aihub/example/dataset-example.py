# 调试镜像环境
# docker run --name aihub -it -v $PWD:/app -v $PWD/../src:/src ccr.ccs.tencentyun.com/cube-studio/aihub:base-python3.9 bash
# 安装python包
# pip uninstall cubestudio
# pip install -U git+https://git.woa.com/tme-kubeflow/aihub.git

import datetime
import json
import os
import random
import shutil
import time
from cryptography.fernet import Fernet
from cubestudio.request.model import Model
from cubestudio.request.model_client import Client,init
from cubestudio.dataset.dataset import Dataset
from cubestudio.dataset.table import InMemoryTable
from cubestudio.dataset.features import Image,Audio,ClassLabel
import pyarrow.compute as pc
import pyarrow as pa
# Import
from pandarallel import pandarallel

# Initialization
pandarallel.initialize(nb_workers=10)


if __name__=="__main__":
    HOST = 'http://data.tme.woa.com'
    HOST = 'http://host.docker.internal'
    token = '在平台右上角个人界面查看Authorization'
    token='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJwZW5nbHVhbiJ9.3WrxmDOK7PMt0P2xdgUx-1HLvhgeNRPKFaeQzFiIkoU'
    init(host=HOST,username='pengluan',token=token)
    key=Fernet.generate_key()
    print(key)
    # key = b'aViHLsGcYgmzMJrS98N2yRD3oTPMZf5JcZvKzr47f6E='

    # 数据集操作
    # datasets = Client(Dataset).search(name="audio_test")
    dataset = Client(Dataset).one(name="audio_test")
    data_dir='audio_test'
    file_path = 'audio_test.zip'
    print(dataset.path)
    # path 用来控制保存到文件
    # features用来显示上传特征列信息
    features = json.dumps(json.load(open('audio_test/audio_test.json')),indent=4,ensure_ascii=False)
    dataset = dataset.update(path='',features=features)
    dataset.compress(file_path,data_dir)
    # dataset.encrypt(file_path,file_path+".crypt",key)
    dataset.upload(file_path)
    # #
    # # 清除下本地模拟使用方
    # os.remove(file_path)
    # os.remove(file_path+".crypt")
    # shutil.rmtree(data_dir)
    # dataset.download()
    # dataset.decrypt(file_path+".crypt",file_path,key)
    # dataset.decompress(file_path,data_dir)
    #
    dataset.load(data_dir)
    table = dataset.table

    # 读取基础属性
    print(table.info)
    print(table.features)

    # dataset索引，会decode_example
    # print(dataset[1])
    # print(dataset['audio'])
    #
    # # table索引
    # # id_column = table['id']      # 可按列索引
    # # mul_column = table[("id","num")]  # 可按列索引
    # # one_row = table[0]  #  索引单行
    # # mmul_row = table[(0,5)]  # 索引单行
    #
    # # 数据列的操作
    # # table = table.add_column(i=3,field_='col4',column=id_column)   # 在指定位置增加列
    # # table = table.append_column(field_='col5',column=id_column)    # 在末尾追加列
    # # table = table.set_column(i=2, field_='col3', column=id_column)
    # # table = table.rename_columns(['id1','audio1','label1'])
    # # table = table.rename_columns({'id':'id1'})
    # # table = table.remove_column(4)
    # # table = table.drop(['col4'])
    #
    # # 数据切片
    # # table = table.select(['id'])   # 选择列
    # # table = table.slice(1,3)       # 选择行
    # # table = table.sort_by([("num", "ascending")])   # 排序
    #
    # # table = table.filter(pc.equal(table["num"], 2))         # 过滤行
    # # table = table.filter(pc.field('num').isin([1, 2, 3]))  # 过滤行
    # # table = table.filter(pc.field("num") <= 2)         # 过滤行
    # # table = table.filter((pc.field("id")=='BAC009S0764W0121') | (pc.field("num")==3))  # 过滤行
    #
    #
    # # 数据合并,变换
    # # table = table.concat_table(table)
    # # table = table.flatten()
    #
    # # 列的的计算操作
    # # col = table['num']
    # # print(pc.mean(col))
    # # print(pc.min_max(col))
    # # print(pc.value_counts(col))
    #
    # # 列的类型变换
    # # table = table.cast_column("audio", Audio(sampling_rate=16000，decode=True))
    # # table = table.cast_column("image", Image())
    # #
    # # # table/dataframe类型转换
    # # df = table.to_pandas()   # feature元信息丢失
    # # table = InMemoryTable(pa.Table.from_pandas(df),features=table.features)
    # # print(table.features)
    #
    # # 实现map函数
    # # 通过pandas apply实现并行
    # # df = table.to_pandas()
    # # sub_df = df.parallel_apply(lambda item:item["num"]+1,axis=1)
    # # print(sub_df)
    # # index=0
    # def prepare_dataset(batch):
    #     audio = batch["audio"]
    #     audio['sampling_rate']='16000'
    #     # print(audio)
    #     # print(type(audio),type(batch))
    #     return batch
    #
    # table = table.map(prepare_dataset)
    # print(type(table))
    # print(table[0])
    # # print(str(table))
    #
    #


