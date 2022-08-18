# download_minio 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/ner:20220812
启动参数：
```bash
{
 "参数分组1": {
        "--model": {
            "type": "str",
            "item_type": "str",
            "label": "参数1",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "模型",
            "placeholder": "",
            "describe": "model",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--objectname": {
            "type": "str",
            "item_type": "str",
            "label": "参数2",
            "require": 1,
            "choice": [
                "resume_BIO.txt",
                "people_daily_BIO.txt"
            ],
            "range": "",
            "default": "resume_BIO.txt",
            "placeholder": "",
            "describe": "训练数据集选择",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--epochs": {
            "type": "str",
            "item_type": "str",
            "label": "参数3",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "1",
            "placeholder": "",
            "describe": "epochs",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--path": {
            "type": "str",
            "item_type": "str",
            "label": "参数4",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/admin/NER/zdata/",
            "placeholder": "",
            "describe": "数据集路径",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "-pp": {
            "type": "str",
            "item_type": "str",
            "label": "参数6",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/admin/model.pkl",
            "placeholder": "",
            "describe": "模型保存目录",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```

## 参考资料：

1. [博客](https://blog.csdn.net/zp563987805/article/details/104562798/?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0--blog-119957026.pc_relevant_paycolumn_v3&spm=1001.2101.3001.4242.1&utm_relevant_index=3)
2. [github](https://github.com/BeHappyForMe/chinese-sequence-ner)