# tensorflow模型训练任务配置（Runner方式）：
内部会把所有的分布式细节都封装起来，用户不需要关心模型代码的分布式改造，只需要定义模型结构和数据解析逻辑。关于runner封装的代码改造说明见[runner封装训练代码编写规范(tensorflow)]()。
``` 
{
    "job_detail": {
    	"script_name": "<str>",
        "model_args": {
            "<str>": <any>,
            ...
        },
        "train_data_args": {
            "<str>": <any>,
            ...
        },
        "val_data_args"：{
            "<str>": <any>,
            ...
        },
        "train_args": {
            "batch_size": <int>,
            "epochs": <int>,
            "num_samples": <int>,
            "num_val_samples": <int>,
            "optimizer": <str|dict>,
            "losses": <str|dict>,
            "metrics": <str|dict> | [<str|dict>],
            "early_stopping": {
            	"monitor": "<str>",
                "min_delta": <float>,
                "patience": <int>,
                "mode": "<str>"
            },
            "save_path": "${DATA_PATH}$/saved_model"
        }
    }
}
```
- **job_detail**：任务的详细描述，其下字段有：
	- **script_name**：用户训练脚本文件名，默认在包目录中，支持相对路径
    - **model_args**：awf_create_model_fn回调函数的参数，关于awf_create_model_fn函数请参考[]。
    - **train_data_args**：awf_create_train_dataset_fn回调函数的参数，关于awf_create_train_dataset_fn请参考[]。
    - **val_data_args**：awf_create_val_dataset_fn回调函数的参数，关于awf_create_val_dataset_fn请参考[]。
    - **train_args**：训练过程的参数。其下字段有：
        - **batch_size**：训练批次大小
        - **epochs**：训练epoch数量
        - **num_samples**：训练集样本数，!!#ff0000 分布式训练（num_workers>1）时必填!!（不必很精确，大致量级能对上即可）。单机情况下不需要填。
		- **num_val_samples**：验证集(test_data_args指定的数据)样本数，!!#ff0000 分布式训练（num_workers>1）且有验证集时!!（不必很精确，大致量级能对上即可）。单机情况或者没有验证集下不需要填。
        - **optimizer**：优化器，可以用字符串指定名字，例如"adam"表示使用adam优化器；也可以用dict指定详细的优化器参数，例如：
            ``` 
            {
                "type": "adam",
                "args": {
                    "learning_rate": 0.001
                }
            }
            ```
          表示使用adam优化器，并且把learning rate设置为0.001。目前支持的优化器见**优化器表**。**如果是使用多优化器，则optimizer可以配置成一个数组。数组中每一个与单个配置一样。数组中的优化器顺序需要与awf_group_trainable_vars_fn回调里面定义的训练参数分组对应**。awf_group_trainable_vars_fn说明见[]，多优化器使用的完整示例见[]。 目前平台还支持使用PCGrad(Project Conflicting Gradients) 改进多任务学习的效果，只需要在参数配置时的dict中加入 `"grad_process": "pcgrad"`即可启用PCGrad，来缓解多任务的梯度被某一个任务的梯度主导的问题。
        - **losses**：指定损失函数，可以是单个或者多个（对应于多输出模型）。同样也可以使用名字或者dict指定详细参数。目前支持的损失函数见支持**损失函数表**。需要注意的是，**如果是多任务训练，即模型有多个输出，同时数据具有多个label，那么指定的loss要么为1个，要么数量必须和任务数量相同，即要求对于每个任务都指定一个loss**；具体的执行中，当指定的loss为1个时，所有的任务都会采用相同的loss，而当指定的loss数量大于1时，loss和多任务的label将会按照顺序一一组合执行，如果任务数多于loss数，那么最后几个多出的label将不会被使用到。
        - **metrics**：指定训练过程的评估指标，可以单个或者多个，对于多输出模型，需要用子列表来指定每个输出的metric，例如`[['auc']， ['mse']]`表示对第一个输出使用auc，第二个输出使用mse。同样也可以使用名字或者dict指定详细参数。目前支持的metric见支持**metric表**。
        - **early_stopping**：!!#ff0000 非必填!!，early stopping插件，便于避免训练过拟合。其下字段有：
        	- **monitor**：early stopping的参考指标值，即上面metric中的某一个，需要注意的是，因为我们判定过拟合一般是使用验证集上的指标来做参考的，所以这里需要在metric名字前加上"val_"前缀，例如"val_auc"，表示监控验证集上的auc值。
            - **min_delta**：判定指标值有提升的最小变化值，指标值提高/减低小于这个值认为指标没有改进。
            - **patience**：停止训练前最多忍受的指标值不改进的epoch数，即如果连续有patience个epoch指标值都没有改进则认为达到early stopping条件。
            - **mode**：max表示指标值越大越好，例如auc；min表示指标值越小越好，例如mse。
		- **tensorboard**：!!#ff0000 非必填!!，tensorboard插件，支持的参数见https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/callbacks/TensorBoard，可以不填参数，默认配置为：{}
		- **train_speed_logger**：!!#ff0000 非必填!!，训练速度打印插件，支持的参数有：
			- **every_batches**：!!#ff0000 非必填!!，多少个batch打印一次速度，建议不要设置太小，以免影响训练速度，默认为100
			- **with_metrics**：!!#ff0000 非必填!!，打印速度时，是否也打印metric值，默认为false。
		- **save_path**：训练完的模型保存目录，默认在数据目录中。默认为`${DATA_PATH}$`/saved_model