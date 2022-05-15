# kaldi分布式训练 模板
镜像：ai.tencentmusic.com/tme-public/kaldi_distributed_on_volcano:v2
挂载：4G(memory):/dev/shm,kubernetes-config(configmap):/root/.kube
环境变量：
```bash
NO_RESOURCE_CHECK=true
TASK_RESOURCE_CPU=4
TASK_RESOURCE_MEMORY=4G
TASK_RESOURCE_GPU=0
```
账号：kubeflow-pipeline
启动参数：
```bash
{
    "shell": {
        "--working_dir": {
            "type": "str",
            "item_type": "str",
            "label": "",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "启动目录",
            "describe": "启动目录",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--user_cmd": {
            "type": "str",
            "item_type": "str",
            "label": "",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "./run.sh",
            "placeholder": "启动命令",
            "describe": "启动命令",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--num_worker": {
            "type": "str",
            "item_type": "str",
            "label": "",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "2",
            "placeholder": "worker数量",
            "describe": "worker数量",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--image": {
            "type": "str",
            "item_type": "str",
            "label": "",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "ai.tencentmusic.com/tme-public/kaldi_distributed_worker:v1",
            "placeholder": "",
            "describe": "worker镜像，直接运行你代码的环境镜像 <a href='https://github.com/tencentmusic/cube-studio/tree/master/images'>基础镜像</a>",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```

# 使用说明

说明：用户通过working_dir参数指定一个工作目录，工作目录需包含run.sh、path.sh等，参照kaldi官方样例 /egs/wsj/s5。同时需要将path里面的 KALDI_ROOT 修改为 /opt/kaldi/。当需要在k8s上运行分布式任务时，将cmd.sh里面的各个pl替换为k8s.pl（跟原来的替换为queue.pl类似），k8s.pl 的参数和 run.pl 一样，不接受限制资源的参数（资源通过模板参数调整）。任务运行时，会以工作目录下run.sh为执行入口。
- **--working_dir**:如上文描述。
- **--num_worker**:启动多少个worker节点。
- **--image**:默认即可，如果要升级kaldi可以更换，需要保证镜像内/opt/kaldi/安装了kaldi，并且安装了ssh工具。