# tensorflow模型离线预测任务配置
``` 
{
    "job_detail": {
        "script_name": "<str>",
        "predict_data_args": {
             "<str>": <any>
		},
        "predict_args": {
            "model_path": "<str>",
            "model_name": "<str>",
            "batch_size": <int>,
            "result_file": "<str>",
            "result_field_delim": "<str>",
            "output_delim": "<str>",
            "row_id_col": "<str>",
            "row_format": "<str>",
            "write_headers": <bool>
        }
    }
}
```
- **job_detail**: 任务的详细描述，其下字段有：
	- **script_name**：用户脚本文件名，默认在包目录中，支持相对路径
	- **predict_data_args**：awf_predict_val_dataset_fn回调函数的参数，关于awf_predict_val_dataset_fn请参考[]
	- **predict_args**：评估流程详细参数，其下字段有：
		- **model_path**：必填，模型路径
		- **model_name**：非必填，模型名字
		- **batch_size**：非必填，预测时的batch大小，不填时默认1024
		- **result_file**：非必填，预测结果文件路径，不填时默认在数据路径下的predict.csv。数据格式为csv，第一行为列名。当是单输出模型时，最后一列为预测结果值，列名为"output"，当是多输出时，每个输出占一列，当模型输出命名时，输出列名是输出名，当不命名时，输出列名是"output_<index>"，<index>是输出序号，从0开始。
		- **result_field_delim**：非必填，预测结果文件的列分隔符，不填时默认为空格。
		- **output_delim**：非必填，预测结果向量维度分割符，不填时默认为逗号，当预测结果是向量（例如embedding）时有用。
		- **write_headers**：非必填，预测结果是否需要写文件头，不填默认为True
		- **row_id_col**：非必填，输出的行标识。row_id_col指定的列必须在输入文件中存在，但不是模型的输入特征。不填时，输出结果中只包含模型输出；如果指定了row_id_col，则输出文件的第一列是输入文件对应列的值。
		- **row_format**：非必填，预测结果每一行的格式，格式的指定的基本单位是"{<key>}"，其中<key>可取值有
			- **row_id**：仅当指定了row_id_col有效，结果文件中{row_id}会被替换为row_id_col指定列的值。另外也可以使用row_id_col值，例如row_id_col="uin"，则{row_id}和{uin}是等价的。
			- **output_\<idx\>**：当模型有多输出时，\<idx\>是输出的序号，从0开始。另外{output}和{output_0}是等价的。对于只有一个输出的模型指定{output}即可。格式化时，同一个{<key>}可以多次出现，每一个{<key>}两边都可以有任意字符串，例如一个输出用户embedding的模型，预测数据文件中有一列uid用于标识用户。那么"{uid} | {uid} | {ouput}"对应的每一行结果就类似于"123 | 123 | 0.1,0.2,0.3"这样。其中123是某一个uid，0.1,0.2,0.3是对应的embedding（这里假设使用默认维度分隔符","）。