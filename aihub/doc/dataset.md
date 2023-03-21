
# 数据集必备资料

dataset_name.csv 包含了整个数据集的内容   (二选一)

dataset_name.py load函数 加载返回pyarrow table (二选一)

dataset_name.json 列格式   (可选)

示例：
```json
{
  "features": {
    "image": {
      "_type": "Image"
    },
    "num": {
      "_type": "Value",
      "dtype": "int32"
    },
    "id": {
      "_type": "Value",
      "dtype": "string"
    },
    "audio": {
      "_type": "Audio"
    },
    "label": {
      "num_classes": 5,
      "names": [
        "class1",
        "class2",
        "class3",
        "class4",
        "class5"
      ],
      "_type": "ClassLabel"
    }
  }
}
```

列格式：

value：整型，浮点，字符串

audio：path/url

image: path/url

label: json 和 具体label和解析函数对应即可


# 数据集搜索/查询/修正/加解密/解压缩/上传下载

    # datasets = Client(Dataset).search(name="audio_test")
    # dataset = Client(Dataset).one(name="audio_test")
    # dataset = dataset.update(path='')  # 如果是全部替换掉就使用
    # dataset.compress(file_path,data_dir)
    # dataset.encrypt(file_path,file_path+".crypt",key)
    # dataset.upload(file_path+".crypt")

    # dataset.download()
    # dataset.decrypt(file_path+".crypt",file_path,key)
    # dataset.decompress(file_path,data_dir)

# 数据集加载索引

    dataset.load(data_dir)
    table = dataset.table

    # 读取基础属性
    print(table.info)
    print(table.features)

    # dataset索引，会decode_example
    # print(dataset[1])
    # print(dataset['audio'])

# table索引

    # id_column = table['id']      # 可按列索引
    # mul_column = table[("id","num")]  # 可按列索引
    # one_row = table[0]  #  索引单行
    # mmul_row = table[(0,5)]  # 索引单行

# 数据列的操作

    # table = table.add_column(i=3,field_='col4',column=id_column)   # 在指定位置增加列
    # table = table.append_column(field_='col5',column=id_column)    # 在末尾追加列
    # table = table.set_column(i=2, field_='col3', column=id_column)
    # table = table.rename_columns(['id1','audio1','label1'])
    # table = table.rename_columns({'id':'id1'})
    # table = table.remove_column(4)
    # table = table.drop(['col4'])

# 数据切片

    # table = table.select(['id'])   # 选择列
    # table = table.slice(1,3)       # 选择行
    # table = table.sort_by([("num", "ascending")])   # 排序

    # table = table.filter(pc.equal(table["num"], 2))         # 过滤行
    # table = table.filter(pc.field('num').isin([1, 2, 3]))  # 过滤行
    # table = table.filter(pc.field("num") <= 2)         # 过滤行
    # table = table.filter((pc.field("id")=='BAC009S0764W0121') | (pc.field("num")==3))  # 过滤行


# 数据合并,变换

    # table = table.concat_table(table)
    # table = table.flatten()

# 列的的计算操作

    # col = table['num']
    # print(pc.mean(col))
    # print(pc.min_max(col))
    # print(pc.value_counts(col))

# 列的类型变换

    # table = table.cast_column("audio", Audio(sampling_rate=16000，decode=True))
    # table = table.cast_column("image", Image())

# table/dataframe类型转换

    # df = table.to_pandas()   # feature元信息丢失
    # table = InMemoryTable(pa.Table.from_pandas(df),features=table.features)
    # print(table.features)

# 并行计算

    # 通过pandas apply实现并行
    # df = table.to_pandas()
    # sub_df = df.parallel_apply(lambda item:item["num"]+1,axis=1)
    # print(sub_df)

    def prepare_dataset(batch):
        audio = batch["audio"]
        audio['sampling_rate']='16000'
        return batch

    table = table.map(prepare_dataset)
    print(table[0])
