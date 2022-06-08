# 模型离线分布式推理 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/volcano:offline-predict-20220101
挂载：kubernetes-config(configmap):/root/.kube
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
            "label": "启动目录",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/xx",
            "placeholder": "",
            "describe": "启动目录",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--command": {
            "type": "str",
            "item_type": "str",
            "label": "环境安装和任务启动命令",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "/mnt/xx/../start.sh",
            "placeholder": "",
            "describe": "环境安装和任务启动命令",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--num_worker": {
            "type": "str",
            "item_type": "str",
            "label": "占用机器个数",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "3",
            "placeholder": "",
            "describe": "占用机器个数",
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
            "default": "ccr.ccs.tencentyun.com/cube-studio/ubuntu-gpu:cuda10.1-cudnn7-python3.6",
            "placeholder": "",
            "describe": "worker镜像，直接运行你代码的环境镜像<a href='https://github.com/tencentmusic/cube-studio/tree/master/images'>基础镜像</a>",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```

# 用户代码示例

## 启动shell脚本
主要包含 环境安装，和并发任务启动部分
```
# 安装包环境
pip install tme-di numpy pandas pysnooper pika psutil pynvml --index-url https://mirrors.cloud.tencent.com/pypi/simple/
# 安装自己需要的环境
pip install xx

# 自定义单worker内并发数量，提高利用率
for index in $(seq 0 2)  
do
{
    export LOCAL_RANK=$index
    python your_code.py
}&;
done
wait
```

## 构建派生类your_code.py
基于Offline_Predict 实现datasource方法和predict方法.

```
import tensorflow as tf
import os
import numpy
from di.cube.offline_predict_model import Offline_Predict

class My_Offline_Predict(Offline_Predict):
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        gpus = tf.config.list_physical_devices('GPU')
        # print(gpus)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        self.model = tf.saved_model.load('/mnt/xx/..')

    # 定义所有要处理的数据源，返回字符串列表
    def datasource(self):
        all_lines = open('/mnt/xx/../all_video_path.txt', mode='r').readlines()
        all_lines = all_lines+all_lines+all_lines+all_lines+all_lines+all_lines+all_lines
        return all_lines

    # 定义一条字符串数据的处理逻辑
    def predict(self,value):
        result = self.model(value)
        # print(result)
        return result

My_Offline_Predict().run()
```
