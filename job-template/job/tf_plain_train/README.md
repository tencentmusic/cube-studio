# tensorflow模型训练任务配置(plain方式)：
用户如果要使用分布式，需要自己对代码做分布式改造，框架仍然会把TF_CONFIG写入环境变量中，用户代码可以通过os.environ.get('TF_CONFIG')获取。这种方式下用户的代码有更大灵活性。
``` 
{
    "job_detail": {
    	"script_name": "<str>",
        "params": [
        	<str|int|float>, ...
        ],
        "model_name": "<str>",
        "save_path": "<str>"
    }
}
```
- **job_detail**：任务的详细描述，其下字段有：
    - **script_name**：**!!#e06666 必填!!**。用户训练脚本文件名，默认在包目录中，支持相对路径
    - **params**：传递给用户脚本的参数，具体参数取决于用户脚本。
    - **model_name**：模型名字。
    - **save_path**：训练完的模型保存目录，默认在数据目录中，支持相对路径。**一般情况下保存目录还会在params传递给用户脚本以便实际执行模型保存操作，这里需要注意两者保持一致。例如params可能包含参数`["--save-dir", "${DATA_PATH}$/saved_model"]`，那么save_path也应该设置为`"${DATA_PATH}$/saved_model"`。**