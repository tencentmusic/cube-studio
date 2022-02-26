# tensorflow模型评估任务配置
``` 
{
        "job_detail": {
            "script_name": "<str>",
            "evaluation_args": {
                "losses": "<str|dict>",
                "batch_size": <int>,
                "metrics": <str|dict>,
                "model_call_exclude_input_index": <int>|[<int>...],
                "input_squeeze": <bool>,
                "models": [
                    {
                        "name": "<str>",
                        "path": "<str>"
                    },
                    ...
                ],
                "output_file": "<str>"
            },
            "test_data_args": {
            	"<str>": <any>
            }
        }
}
```
- **job_detail**: 任务的详细描述，其下字段有：
	- **script_name**：用户脚本文件名，默认在包目录中，支持相对路径
    - **evaluation_args**：评估流程详细参数，其下字段有：
    	- **losses**：用法与训练时一样。
        - **batch_size**：评估数据的批次大小
        - **metrics**：用法与训练时一样，但是可以使用与训练时不同的metric
        - **model_call_exclude_input_index**：当模型数据集有多输入时，call模型时是否要排除某些输入。一般情况下不用设置这个参数，目前唯一的使用场景是使用gauc metric时，因为metric输入中需要加上用户标识uid，而在模型训练或者预测时这个uid又是用不到的，这样在Dataset输出和模型输入上就有了差异，此时可以通过设置这个参数来修改call模型时候的输入。
        - **input_squeeze**：与model_call_exclude_input_index类似，一般情况下也用不到，主要是标识如果call模型输入中去掉了model_call_exclude_input_index设置的几路输入后只剩下一路，是否需要去掉输入外层的tuple或list，类似于张量处理中的squeeze压缩。
        - **models**：需要作为对比的模型数组，其中每个模型有如下字段（!!#cc0000 **有多个模型对比时，最后一个模型将被视为基线模型**!!）：
        	- **name**：模型名称
            - **path**：模型文件所在目录，可以使用`${ONLINE_MODEL}$`魔法变量
            **!!#ff00ff 注意所有模型应该具有相同的输入和输出，另外models数组中的最后一个模型将被认为是基线模型!!**
		- **output_file**：评估结果输出文件名字（!!#cc0000 不要包含路径名，输出文件会放在数据目录下!!），文件内容是一个json dict格式。其中包含所有模型的评估结果，**!!#cc0000 一般如果下游要接部署任务的话，可以配置output_file，并把这个文件名配置为部署任务的upstream_output_file，这样部署任务就会从这个文件里面获得最优模型的路径。!!**
        - **test_data_args**：awf_create_test_dataset_fn回调函数的参数，关于awf_create_test_dataset_fn请参考[]。
