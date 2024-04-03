# deploy-service 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/deploy-service:20211001
启动参数：
```bash
{
    "模型信息": {
        "--label": {
            "type": "str",
            "item_type": "str",
            "label": "中文描述描述",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "demo推理服务",
            "placeholder": "",
            "describe": "推理服务描述",
            "editable": 1
        },
        "--model_name": {
            "type": "str",
            "item_type": "str",
            "label": "模型名",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型名",
            "editable": 1
        },
        "--model_version": {
            "type": "str",
            "item_type": "str",
            "label": "模型版本号",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "v2022.10.01.1",
            "placeholder": "",
            "describe": "模型版本号",
            "editable": 1
        },
        "--model_path": {
            "type": "str",
            "item_type": "str",
            "label": "模型地址",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型地址",
            "editable": 1
        }
    },
    "部署信息": {
        "--service_type": {
            "type": "str",
            "item_type": "str",
            "label": "推理服务类型",
            "require": 1,
            "choice": [
                "serving",
                "ml-server",
                "tfserving",
                "torch-server",
                "onnxruntime",
                "triton-server",
                "llm-server"
            ],
            "range": "",
            "default": "service",
            "placeholder": "",
            "describe": "推理服务类型",
            "editable": 1
        },
        "--images": {
            "type": "str",
            "item_type": "str",
            "label": "推理服务镜像",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "推理服务镜像",
            "editable": 1
        },
        "--working_dir": {
            "type": "str",
            "item_type": "str",
            "label": "推理容器工作目录",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "推理容器工作目录,个人工作目录/mnt/$username",
            "editable": 1
        },
        "--command": {
            "type": "str",
            "item_type": "str",
            "label": "推理容器启动命令",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "推理容器启动命令",
            "editable": 1
        },
        "--env": {
            "type": "text",
            "item_type": "str",
            "label": "推理容器环境变量",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "推理容器环境变量",
            "editable": 1
        },
        "--ports": {
            "type": "str",
            "item_type": "str",
            "label": "推理容器暴露端口",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "80",
            "placeholder": "",
            "describe": "推理容器暴露端口",
            "editable": 1
        },
        "--replicas": {
            "type": "str",
            "item_type": "str",
            "label": "pod副本数",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "1",
            "placeholder": "",
            "describe": "pod副本数",
            "editable": 1
        },
        "--resource_memory": {
            "type": "str",
            "item_type": "str",
            "label": "每个pod占用内存",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "2G",
            "placeholder": "",
            "describe": "每个pod占用内存",
            "editable": 1
        },
        "--resource_cpu": {
            "type": "str",
            "item_type": "str",
            "label": "每个pod占用cpu",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "2",
            "placeholder": "",
            "describe": "每个pod占用cpu",
            "editable": 1
        },
        "--resource_gpu": {
            "type": "str",
            "item_type": "str",
            "label": "每个pod占用gpu",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "0",
            "placeholder": "",
            "describe": "每个pod占用gpu",
            "editable": 1
        }
    }
}
```
要想运行该模板任务时自动发布上线，除了可以在host参数中配置域名，还可以在模板的环境变量中配置，这样能通过环境变量的形式主动告诉任务当前浏览器使用的ip
HOST=http://xx.xx.xx.xx