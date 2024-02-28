# datax 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/datax  

参数为job.json文件，格式可参考[datax github官网](https://github.com/alibaba/DataX)

参数
```bash
{
    "参数": {
        "-f": {
            "type": "str",
            "item_type": "str",
            "label": "job.json文件地址，<a target='_blank' href='https://github.com/alibaba/DataX'>书写格式参考</a>",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/usr/local/datax/job/job.json",
            "placeholder": "",
            "describe": "job.json文件地址，<a target='_blank' href='https://github.com/alibaba/DataX'>书写格式参考</a>",
            "editable": 1
        }
    }
}
```
