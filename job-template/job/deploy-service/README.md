# deploy-service 模板
镜像：ccr.ccs.tencentyun.com/cube-studio/deploy-service:20240601
启动参数：
```bash
{
    "模型信息": {
        "--project_name": {
            "type": "str",
            "item_type": "str",
            "label": "项目组名称",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "public",
            "placeholder": "",
            "describe": "项目组名称",
            "editable": 1
        },
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
                "vllm"
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
        "--host": {
            "type": "str",
            "item_type": "str",
            "label": "部署域名",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "部署域名，留空自动生成",
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
        },
        "--volume_mount": {
            "type": "str",
            "item_type": "str",
            "label": "挂载",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "kubeflow-user-workspace(pvc):/mnt",
            "placeholder": "",
            "describe": "容器的挂载，支持pvc/hostpath/configmap三种形式,格式示例:$pvc_name1(pvc):/$container_path1,$hostpath1(hostpath):/$container_path2,注意pvc会自动挂载对应目录下的个人username子目录",
            "editable": 1
        },
        "--inference_config": {
            "type": "text",
            "item_type": "str",
            "label": "配置文件",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "会配置文件的形式挂载到容器/config/目录下。<font color='#FF0000'>留空时将被自动重置</font>，格式：<br>---文件名<br>多行文件内容<br>---文件名<br>多行文件内容",
            "editable": 1
        },
        "--metrics": {
            "type": "str",
            "item_type": "str",
            "label": "指标采集接口",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "请求指标采集，配置端口+url，示例：8080:/metrics",
            "editable": 1
        },
        "--health": {
            "type": "str",
            "item_type": "str",
            "label": "健康检查",
            "require": 0,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "健康检查接口，使用http接口或者shell命令，示例：8080:/health或者 shell:python health.py",
            "editable": 1
        }
    }
}
```
要想运行该模板任务时自动发布上线，除了可以在host参数中配置域名，还可以在模板的环境变量中配置，这样能通过环境变量的形式主动告诉任务当前浏览器使用的ip
HOST=http://xx.xx.xx.xx