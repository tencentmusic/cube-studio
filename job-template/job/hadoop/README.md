功能：用于对接公司已存在的大数据平台，此模板需要自行封装大数据组件客户端。 
自行添加dockerfile中spark、hadoop、sqoop、hbase等基础组件客户端。和其他公司基础组件配置或环境

镜像：ccr.ccs.tencentyun.com/cube-studio/hadoop:20221010

参数
```bash
{
    "参数": {
        "--command": {
            "type": "str",
            "item_type": "str",
            "label": "执行命令",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "spark-submit xx",
            "placeholder": "",
            "describe": "执行命令",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```