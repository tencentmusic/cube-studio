# 模板说明

运行单机脚本，支持shell，python脚本

# 模板镜像

`ai.tencentmusic.com/tme-public/python_data_transform:20201010`

# 模板注册
参考上级目录的readme.md，注册时填写以下配置。

1、启动参数：
```
{
    "script_type": "<str>",
    "script_name"："<str>",
    "params": [
    	<str|int|float>,
        ...
    ],
    "export_files": [
    	{
            "tar_file": "<str>",
            "label": "<str>"
        },
        ...
    ]
}
```

    script_type: 必填。脚本类型，目前支持python和shell两种。
    script_name：必填。用户脚本文件名，默认在包目录中，支持相对路径。
    params：非必填。传递给脚本的参数数组，参数支持使用魔法变量。
    export_files：非必填。指定本脚本的输出文件数组，数组每个元素指定一个输出文件。其下字段有：
        tar_file：输出文件名字，默认在数据目录中，支持相对路径。
        label：用户自定义标签。可不填。

# 使用方法
略
