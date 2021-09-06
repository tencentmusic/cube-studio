# 模板说明
文件分割

# 模板镜像

`ai.tencentmusic.com/tme-public/ray:gpu-20210601`

# 模板注册
参考上级目录的readme.md，注册时填写以下配置。

1、启动参数：
```
{
    "source_file"：<str>,
    "source_type": <str>,
    "csv_delimiter": <str>,
    "split_num": <str>,
    "tar_path": <str>,
    "name_prefix": <str>,
    "header": <bool>,
    "delete_source": <bool>
}
```
    source_file： 必填，源文件，支持通配符
    source_type：非必填，源文件类型，目前暂时只支持"csv"，默认"csv"。
    csv_delimiter：必填，csv文件的列分隔符
    split_num：必填，要分割成的文件个数
    tar_path：非必填，分割之后的文件存放路径，如果不填，则默认与source_file在同一个目录下
    name_prefix：非必填，分割之后文件名前缀，如果设置了，则结果文件名为<name_prefix>-part-<id>.<ext>，如果没有设置，则结果文件名为<source_name>-part-<id>.<ext>，其中<id>是文件编号，从0到split_num，<ext>是源文件的扩展名，<source_name>是源文件的除去路径和扩展名之后的部分。
    header：非必填，csv文件是否包含文件头，如果包含文件，分割后的文件也会都包含文件头。默认为true
    delete_source：非必填，分割完之后，是否删掉源文件。默认为true

# 使用方法
略
