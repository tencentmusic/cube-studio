# tensorflow分布式模型训练任务配置：
``` 
{
    "num_workers": <int>,
    "node_affin": "<str>",
    "pod_affin": "<str>",
    "timeout": "<str>",
    "trainer": "<str>",
    "resources": {
        "limits" {
            "cpu": "<str>",
            "memory": "<str>"
        }
    },
    "job_detail": {
        /*
        取决于trainer的配置，runner和plain有不同的配置方式，具体配置请参看对应的文档。
        */
    }
}
```
- **num_workers**：!!#ff0000 非必填!!，分布式训练的节点数，默认为1（即单机）
- **node_affin**：节点选择偏好，有如下几种取值：
    - only_gpu：只使用GPU机器，如果没有可用的，则任务进入排队或调度失败
    - only_cpu：只使用CPU机器，如果没有可用的，则任务进入排队或调度失败
- **pod_affin**：worker分布偏好，有如下几种取值：
	- spread：尽量分散开到不同物理机
    - concent：尽量聚集到相同物理机上
- **timeout**：!!#ff0000 非必填!!，任务超时时间，运行超过这个时间后认为任务失败。格式为“数字+单位”，单位支持毫秒(ms)，秒(s)、分(m)、小时(h)、天(d)、星期(w)。例如"100d"表示100天。默认“365d”
- **trainer**：训练模式，有如下两种取值：
	- **runner**：runner封装方式，内部会把所有的分布式细节都封装起来，用户不需要关心模型代码的分布式改造，只需要定义模型结构和数据解析逻辑。关于runner封装的代码改造说明见[runner封装训练代码编写规范(tensorflow)]()。**建议都使用runner**
    - **plain**：直接调用用户模型脚本，如果要用分布式，需要用户自己做分布式改造，不过这种方式对用户的灵活性更大一些。不一定使用keras api。
- **resources**：!!#ff0000 非必填!!，每个worker的资源大小，支持的配置选项有：
	- **cpu**：cpu核数
	- **memory**：内存大小
	- **nvidia.com/gpu**：gpu个数。大于1时表示单机多卡。
	    - 当node_affin为only_cpu时，默认配置为：
        ``` 
        {
            "limits": {
                "cpu": "16", 
                "memory": "16G"
            }
        }
        ```
        - 当node_affin为only_gpu时，默认配置为：
        ``` 
        {
            "limits": {
                "cpu": "16", 
                "memory": "16G",
                "nvidia.com/gpu": 1,
            }
        }
        ```