DTower算法模板配置示例如下（可以先阅读[《pipeline各任务配置》文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117)中的"tensorflow模型训练任务配置.runner封装方式"一节了解模型训练任务的一般配置方法）:  
  
```   
{  
    "num_workers": 1,  
    "node_affin": "only_cpu",  
    "pod_affin": "spread",  
    "timeout": "100d",  
    "job_detail": {  
        "model_input_config_file": "${PACK_PATH}$/model_input.json",  
        "model_args": {  
            "user_tower_units": [128, 64, 32],  
            "user_tower_hidden_act": ["lrelu", "tanh"],  
            "user_tower_output_act": "sigmoid",  
            "user_tower_dropout": [0, 0.5, 0],  
            "user_tower_use_bn": [true, false],  
            "user_tower_l1_reg": 0.0,  
            "user_tower_l2_reg": 0.0,  
            "item_tower_units": [64, 32],  
            "item_tower_hidden_act": ["lrelu"],  
            "item_tower_output_act": "sigmoid",  
            "item_tower_dropout": 0,  
            "item_tower_use_bn": false,  
            "item_tower_l1_reg": 0.0,  
            "item_tower_l2_reg": 0.0,  
            "pairwise": true,  
            "use_cosine": true,  
            "name": "dtower"  
        },  
        "train_data_args": {  
            "data_file": "${DATA_PATH}$/train-data.csv",  
            "field_delim": " "  
        },  
        "val_data_args": {  
            "data_file": "${DATA_PATH}$/val-data.csv",  
            "field_delim": " "  
        },  
        "user_pred_data_args": {  
           "model_input_config_file": "${PACK_PATH}$/user_pred_input_demo.json",  
           "data_file": "${DATA_PATH}$/user-pred-data-1000.csv",  
           "field_delim": " "  
       },  
       "item_pred_data_args": {  
           "model_input_config_file": "${PACK_PATH}$/item_pred_input_demo.json",  
           "data_file": "${DATA_PATH}$/pred-data-1000.csv",  
           "field_delim": " "  
       },  
        "train_args": {  
            "batch_size": 1024,  
            "epochs": 1,  
            "num_samples": 1000000,  
            "optimizer": {  
                "type": "adam",  
                "args": {  
                    "learning_rate": 0.001  
                }  
            },  
            "losses": "bpr",  
            "metrics": ["po_acc"],  
            "early_stopping": {  
                "monitor": "po_acc",  
                "mode": "max",  
                "patience": 5,  
                "min_delta": 0.001  
            },  
            "train_speed_logger": {"every_batches": 1, "with_metrics": true},  
            "tensorboard": {}  
        },  
        "eval_args": {  
            "batch_size": 256,  
            "metrics": ["po_acc"]  
        },  
        "user_predict_args": {  
            "model_path": "${DATA_PATH}$/saved_model/dtower-user_tower",  
            "result_file": "user_embeddings.txt",  
            "row_id_col": "uid",  
            "output_delim": ",",  
            "row_format": "{uid}|{uid}|{output}"  
        },  
        "item_predict_args": {  
            "model_path": "${DATA_PATH}$/saved_model/dtower-item_tower",  
            "result_file": "item_embeddings.txt",  
            "row_id_col": "item_id",  
            "output_delmi": ",",  
            "row_format": "{item_id}|{item_id}|{output}"  
        }  
    }  
}  
```  
上面配置中，主要关注 **model_input_config_file** 和 **model_args** 中的内容.  
关于输入数据的配置见[算法模板数据配置说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001865665)  
 **model_input_config_file** ：指定了模型输入描述文件，输入描述文件的说明见[算法模板输入描述文件格式说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001851927)  
 **load_model_from** ：可以指定模型加载路径，如果配置了这个参数，且路径真是存在，则会跳过训练过程，从路径加载模型。这个主要是用于不想训练，只是想用评估数据跑下之前训好模型的指标的场景（配置了val_data_args和eval_args）  
 **model_args** 中是特定于DTower模型的参数：  
 -  **user_tower_units** ：!!#ff0000 必填!!。用户塔DNN的网络结构，数组中每一个元素是网络对应层的宽度，**!!#990000 从下至上!!**。网络最后一层宽度即user embedding的维度。**!!#cc0000 注意用户塔DNN输出层维度和item塔DNN输出层维度必须保持一致。!!**  
 - **user_tower_hidden_act**：!!#ff0000 非必填!!。用户塔DNN的隐层（即除最后输出层的其他层）的激活函数类型，如果设置为一个字符串，例如"relu"，则表示所有隐层的激活函数都是一样的；!!#e06666 如果需要每层指定不同的激活函数，则可以设置为数组，例如["relu", "sigmoid"]，表示第一层使用relu，第二层使用sigmoid，层顺序与 *user_tower_units* 一致。设置的数组长度不一定与隐层数量一样，当 *user_tower_hidden_act* 长度超过隐层数量时，超出的部分会被自动忽略，如果比隐层数量短，则缺少部分对应的隐层就没有激活函数。如果是某个中间位置隐层不需要激活函数，而上下都需要，则对应位置设置为"none"即可，例如有三个隐层，其中第一和第三层分别是relu和sigmoid，而第二层不需要，则可以设置为["relu", "none", "sigmoid"]。!!  
 - **user_tower_output_act**：!!#ff0000 非必填!!。用户塔DNN的输出层激活函数。  
 - **user_tower_dropout**：!!#ff0000 非必填!!。用户塔DNN隐层的dropout的概率，如果设置成一个值，例如0.5，则表示所有隐层都采用0.5的dropout；如果需要每层采用不同的dropout值，则可以设置为数组，例如[0.5, 0.2]，表示第一层采用0.5的dropout，第二层采用0.2的dropout。与*user_tower_hidden_act*类似，数组长度也不一定与隐层数量一样，长度不一样时，对应关系与*user_tower_hidden_act*中说明一样。  
 - **user_tower_use_bn**：!!#ff0000 非必填!!。用户塔DNN各层（包含输出层）是否使用batch normalization。如果设置成一个值，例如true，则表示所有层都使用batch normalize；如果各层不一样，则可以设置成数组，例如[true, false]，表示第一层使用BN，第二层不使用。与user_tower_hidden_act类似，数组长度也不一定与隐层数量一样，长度不一样时，对应关系与user_tower_hidden_act中说明一样。  
 - **user_tower_l1_reg**：!!#ff0000 非必填!!。用户塔DNN的参数L1正则化系数。  
 - **user_tower_l2_reg**：!!#ff0000 非必填!!。用户塔DNN的参数L2正则化系数。  
 - **user_tower_use_bias**：!!#ff0000 非必填!!。用户塔DNN的是否使用bias。默认为false  
 - **item_tower_***： item塔DNN的相关配置，与user塔DNN对应参数用法一样。**!!#cc0000 注意用户塔DNN输出层维度和item塔DNN输出层维度必须保持一致。!!**  
 - **pairwise**：!!#ff0000 非必填!!。是否使用pairwise方式，目前双塔模型支持pointwise和pairwise两种方式。默认是false，即pointwise的。使用pairwise和pointwise在配置上有一些差异，如下表所述：  
  
| | 训练样本 | loss | metric |  
| ------ | ------ | ------ | ------ |  
| pair wise | 每一行样本包含**用户特征列**，**正item特征列**，**负item特征列**三个部分，其中正负item的列数量和类型应该是一样的（列名不一样），且**没有label列**。在模型输入描述中需要对特征分组，用户特征组叫**user**，正样本特征组叫**p_item**，负样本特征组叫**n_item**。 | 目前支持**bpr**和**pair_hinge**两种loss，具体请查看[loss列表](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001868155) | 目前支持**po_acc** metric，具体请查看[metric列表](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001946487) |  
| point wise | 每一行样本包含**用户特征列**，**样本item特征列**，**label**三个部分。在模型输入描述中需要对特征分组，用户特征组叫**user**，item特征组叫**item**，label列单独一组，组名无要求。 | 无特殊要求，与其他CTR模型一样 | 无特殊要求，与其他CTR模型一样 |  
  
 - **use_cosine**：!!#ff0000 非必填!!。user/item embedding是否做归一化。默认为false。  
 -  **name** ：!!#ff0000 非必填!!。模型名字，可以自定义。默认为"dtower"。  
  
其他参数的意义都可以参考《pipeline各任务配置》文档。  
pipeline全流程的使用说明见[模型开发使用文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001727011)  
  
**双塔模型训练完成之后，会在模型保存目录下生成四个子目录：<model_name>，<model_name>-user_tower，<model_name>-item_tower，<model_name>-ctr_model。分别对应完整双塔模型，用户塔模型，item塔模型，ctr预估模型（这里只是命名叫ctr，具体含义由业务数据决定）。用户塔、item塔、ctr预估模型都可以单独用作serving，用户塔模型输入与模型输入描述中user特征组部分一致；item塔模型输入在pairwise方式下与p_item特征组一致，pointwise方式下与item特征组一致；ctr预估模型在pairwise方式下输入包含user特征组和p_item特征组特征，在pointwise方式下包含user特征组和item特征组特征。**  
  
双塔算法模板支持用户向量和item向量的离线预测导出，相关配置如下：  
- **user_pred_data_args**：用户预测数据配置，具体数据配置方法见[算法模板数据配置说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001865665)  
- **item_pred_data_args**：item预测数据配置，具体数据配置方法见[算法模板数据配置说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001865665)  
- **user_predict_args**：用户预测相关配置选项，具体配置方法见[tensorflow模型离线预测任务配置](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc13)中关于predict_args参数的说明  
- **item_predict_args**：item预测相关配置选项，具体配置方法见[tensorflow模型离线预测任务配置](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc13)中关于predict_args参数的说明