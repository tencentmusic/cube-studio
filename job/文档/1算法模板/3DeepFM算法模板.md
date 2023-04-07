DeepFM算法模板配置示例如下（可以先阅读[《pipeline各任务配置》文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117)中的"tensorflow模型训练任务配置.runner封装方式"一节了解模型训练任务的一般配置方法）:  
  
```   
{  
    "num_workers": 1,  
    "node_affin": "only_cpu",  
    "pod_affin": "spread",  
    "timeout": "100d",  
    "job_detail": {  
        "model_input_config_file": "${PACK_PATH}$/deepfm_test/model_input_demo.json",  
        "model_args": {  
            "k": 32,  
            "dnn_hidden_layers": [128, 64],  
            "dnn_hidden_act_fn": "relu",  
            "name": "deepfm"  
        },  
        "train_data_args": {  
            "data_file": "${DATA_PATH}$/train_data/*.csv",  
            "file_type": "csv",  
            "field_delim": " "  
        },  
        "val_data_args": {  
            "data_file": "${DATA_PATH}$/val_data/*.csv",  
            "file_type": "csv",  
            "field_delim": " "  
        },  
        "predict_data_args": {  
            "model_input_config_file": "${PACK_PATH}$/deepfm_test/model_predict_input_demo.json",  
            "data_file": "${DATA_PATH}$/val_data/*.csv",  
            "file_type": "csv",  
            "field_delim": " "  
        },  
        "train_args": {  
            "batch_size": 512,  
            "epochs": 1,  
            "num_samples": 1000,  
            "validation_steps": 1,  
            "optimizer": {  
                "type": "adam",  
                "args": {  
                    "learning_rate": 0.001  
                }  
            },  
            "losses": "bce",  
            "metrics": ["auc"],  
            "early_stopping": {  
                "monitor": "mse",  
                "mode": "min",  
                "patience": 1,  
                "min_delta": 0.001  
            },  
            "train_speed_logger": {"every_batches": 100, "with_metrics": true}  
        },  
        "eval_args": {  
            "batch_size": 256,  
            "metrics": [  
                ["bce", "auc"]  
            ]  
        },  
        "predict_args": {  
        }  
    }  
}  
```  
  
上面配置中，主要关注**model_input_config_file**和**model_args**中的内容.  
关于输入数据的配置见[算法模板数据配置说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001865665)  
**model_input_config_file**：指定了模型输入描述文件，输入描述文件的说明见[算法模板输入描述文件格式说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001851927)  
**load_model_from**：可以指定模型加载路径，如果配置了这个参数，且路径真是存在，则会跳过训练过程，从路径加载模型。这个主要是用于不想训练，只是想用评估数据跑下之前训好模型的指标的场景（配置了val_data_args和eval_args）  
**model_args**中是特定于DeepFM模型的参数：  
- **k**：!!#e06666 必填!!。FM侧中使用隐向量维度大小  
- **dnn_hidden_layers**：!!#e06666 必填!!。深度侧的隐层网络结构，是一个整数数组，从左到右对应dnn从底向上每一个隐层的宽度（不包括输入和输出层）。如果是一个空数组，则表示不使用DNN，此时DeepFM退化为FM模型。  
- **dnn_hidden_act_fn**：!!#e06666 非必填!!。DNN隐层使用的激活函数，!!#e06666 如果设置为一个字符串，例如"relu"，则表示所有隐层的激活函数都是一样的；如果需要每层指定不同的激活函数，则可以设置为数组，例如["relu", "sigmoid"]，表示第一层使用relu，第二层使用sigmoid，层顺序与  *dnn_hidden_layers*  一致。设置的数组长度不一定与隐层数量一样，当  *dnn_hidden_act_fn*  长度超过隐层数量时，超出的部分会被自动忽略，如果比隐层数量短，则缺少部分对应的隐层就没有激活函数。如果是某个中间位置隐层不需要激活函数，而上下都需要，则对应位置设置为"none"即可，例如有三个隐层，其中第一和第三层分别是relu和sigmoid，而第二层不需要，则可以设置为["relu", "none", "sigmoid"]。!!  
- **dnn_l1_reg**：!!#e06666 非必填!!。DNN侧的参数L1正则化系数，默认为None。  
- **dnn_l2_reg**：!!#e06666 非必填!!。DNN侧的参数L2正则化系数，默认为None。  
- **dnn_dropout**：!!#e06666 非必填!!。DNN侧训练时dropout的概率大小，!!#e06666 如果设置成一个值，例如0.5，则表示所有隐层都采用0.5的dropout；如果需要每层采用不同的dropout值，则可以设置为数组，例如[0.5, 0.2]，表示第一层采用0.5的dropout，第二层采用0.2的dropout。与 *dnn_hidden_act_fn* 类似，数组长度也不一定与隐层数量一样，长度不一样时，对应关系与 *dnn_hidden_act_fn* 中说明一样。!!  
- **dnn_use_bn**：!!#e06666 非必填!!。DNN侧是否使用batch normalization，!!#e06666 如果设置成一个值，例如true，则表示所有层都使用batch normalize；如果各层不一样，则可以设置成数组，例如[true, false]，表示第一层使用BN，第二层不使用。与 *user_tower_hidden_act* 类似，数组长度也不一定与隐层数量一样，长度不一样时，对应关系与 !!*user_tower_hidden_act* 中说明一样。  
- **embedding_l1_reg**：!!#e06666 非必填!!。FM侧隐向量的L1正则化系数，默认为None。  
- **embedding_l2_reg**：!!#e06666 非必填!!。FM侧隐向量的L2正则化系数，默认为None。  
- **name**：!!#e06666 非必填!!。模型名字，可以自定义。默认为"deepfm".  
  
**predict_args**配置参考[tensorflow模型离线预测任务配置](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc13)，!!#ff0000 **注意模板中script_name参数和predict_args中的model_path参数都不是必填的。**!!  
  
其他参数的意义都可以参考《pipeline各任务配置》文档。  
  
pipeline全流程的使用说明见[模型开发使用文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001727011)