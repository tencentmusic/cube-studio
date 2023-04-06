MMoE算法模板配置示例如下（可以先阅读[《pipeline各任务配置》文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117)中的"tensorflow模型训练任务配置.runner封装方式"一节了解模型训练任务的一般配置方法）:  
  
```   
{  
    "num_workers": 1,  
    "node_affin": "only_cpu",  
    "pod_affin": "spread",  
    "timeout": "100d",  
    "job_detail": {  
        "model_input_config_file": "${PACK_PATH}$/model_input.json",  
        "model_args": {  
            "name": "mmoe_model",  
            "task_structs": [  
                [  
                    32,  
                    8,  
                    1  
                ],  
                [  
                    32,  
                    8,  
                    1  
                ]  
            ],  
            "task_use_bias": true,  
            "task_hidden_act": "relu",  
            "task_output_act": "sigmoid",  
            "task_use_bn": false,  
            "task_l2_reg": 0.001,  
            "num_experts": 3,  
            "expert_layers": [  
                128,  
                64  
            ],  
            "expert_use_bias": true,  
            "expert_l2_reg": 0.001,  
            "gate_use_bias": true  
        },  
        "train_data_args": {  
            "data_file": "${PACK_PATH}$/train_data_trans_norm.csv",  
            "field_delim": ",",  
            "with_headers": false  
        },  
        "val_data_args": {  
            "data_file": "${PACK_PATH}$/test_data_trans_norm.csv",  
            "field_delim": ",",  
            "with_headers": false  
        },  
        "train_args": {  
            "batch_size": 256,  
            "epochs": 20,  
            "num_samples": 100000,  
            "validation_steps": 1,  
            "optimizer": {  
                "type": "adam",  
                "args": {  
                    "learning_rate": 0.001  
                }  
            },  
            "losses": "bce",  
            "metrics": [  
                [  
                    "auc",  
                    "bacc"  
                ],  
                [  
                    "auc",  
                    "bacc"  
                ]  
            ],  
            "early_stopping": {  
                "monitor": "output_2_auc_1",  
                "mode": "max",  
                "patience": 3,  
                "min_delta": 0.001  
            }  
        },  
        "eval_args": {  
            "batch_size": 256,  
            "metrics": [  
                [  
                    "auc",  
                    "bacc"  
                ],  
                [  
                    "auc",  
                    "bacc"  
                ]  
            ]  
        }  
    }  
}  
```  
  
上面配置中，主要关注**model_input_config_file**和**model_args**中的内容.  
关于输入数据的配置见[算法模板数据配置说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001865665)  
**model_input_config_file**：指定了模型输入描述文件，输入描述文件的说明见[算法模板输入描述文件格式说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001851927)  
**load_model_from**：可以指定模型加载路径，如果配置了这个参数，且路径真是存在，则会跳过训练过程，从路径加载模型。这个主要是用于不想训练，只是想用评估数据跑下之前训好模型的指标的场景（配置了val_data_args和eval_args）  
**model_args**中是特定于MMoE模型的参数：  
- **task_structs**：!!#e06666 必填!!。task tower的网络结构数组，数组中每一个元素是一个task的结构，即task dnn每一层的宽度，从下至上。（!!#ea9999 这里需要注意task的顺序与样本中label的顺序的对应!!）  
- **num_experts**：!!#e06666 必填!!。expert的个数。  
- **expert_layers**：!!#e06666必填!!。expert的网络结构。即expert dnn每一层的宽度，从下至上。  
- **expert_use_bias**：!!#e06666非必填!!。expert网络计算线性变换时是否要使用bias，默认为true。  
- **expert_act**：!!#e06666 非必填!!。expert网络的激活函数，!!#e06666 如果设置为一个字符串，例如"relu"，则表示所有隐层的激活函数都是一样的；如果需要每层指定不同的激活函数，则可以设置为数组，例如["relu", "sigmoid"]，表示第一层使用relu，第二层使用sigmoid，层顺序与  *expert_layers*  一致。设置的数组长度不一定与隐层数量一样，当  *expert_act*  长度超过隐层数量时，超出的部分会被自动忽略，如果比隐层数量短，则缺少部分对应的隐层就没有激活函数。如果是某个中间位置隐层不需要激活函数，而上下都需要，则对应位置设置为"none"即可，例如有三个隐层，其中第一和第三层分别是relu和sigmoid，而第二层不需要，则可以设置为["relu", "none", "sigmoid"]。!!  
- **expert_dropout**：!!#e06666 非必填!!。expert网络的层间dropout概率，!!#e06666 如果设置成一个值，例如0.5，则表示所有隐层都采用0.5的dropout；如果需要每层采用不同的dropout值，则可以设置为数组，例如[0.5, 0.2]，表示第一层采用0.5的dropout，第二层采用0.2的dropout。与 *expert_act* 类似，数组长度也不一定与隐层数量一样，长度不一样时，对应关系与 *expert_act* 中说明一样。!!  
- **expert_use_bn**：!!#e06666 非必填!!。expert网络是否使用batch normalization，!!#e06666 如果设置成一个值，例如true，则表示所有层都使用batch normalize；如果各层不一样，则可以设置成数组，例如[true, false]，表示第一层使用BN，第二层不使用。与 *expert_act* 类似，数组长度也不一定与隐层数量一样，长度不一样时，对应关系与 *expert_act* 中说明一样。!!  
- **expert_l1_reg**：!!#e06666 非必填!!。expert网络的参数L1正则化系数，0或者None表示不需要正则化，默认为None。  
- **expert_l2_reg**：!!#e06666 非必填!!。expert网络的参数L2正则化系数，0或者None表示不需要正则化，默认为None。  
- **task_use_bias**：!!#e06666 非必填!!。task网络计算线性变换时是否要使用bias，默认为true。  
- **task_hidden_act**：!!#e06666 非必填!!。task网络隐层的激活函数，配置方式与*expert_act* 一样。  
- **task_output_act**：!!#e06666 非必填!!。task网络输出层的激活函数，默认为None。  
- **task_use_bn**：!!#e06666 非必填!!。task网络是否使用batch normalization，配置方式与*expert_use_bn* 一样。  
- **task_dropout**：!!#e06666 非必填!!。task网络的层间dropout概率，0或None表示不使用dropout，配置方式与*expert_dropout* 一样。  
- **task_l1_reg**：!!#e06666 非必填!!。task网络的参数L1正则化系数，0或者None表示不需要正则化，默认为None。  
- **task_l2_reg**：!!#e06666 非必填!!。task网络的参数L2正则化系数，0或者None表示不需要正则化，默认为None。  
- **gate_use_bias**：!!#e06666 非必填!!。gate网络计算线性变换时是否要使用bias，默认为true。  
- **gate_l1_reg**：!!#e06666 非必填!!。gate网络的参数L1正则化系数，0或者None表示不需要正则化，默认为None。  
- **gate_l2_reg**：!!#e06666 非必填!!。gate网络的参数L2正则化系数，0或者None表示不需要正则化，默认为None。  
- **share_gates**：!!#e06666 非必填!!。不同task是否共享同一个gate，默认为false。如果设置为true，则MMoE模型退化为MoE模型。  
- **name**：!!#e06666 非必填!!。模型名字，可以自定义。默认为"mmoe"。  
  
其他参数的意义都可以参考《pipeline各任务配置》文档。  
  
pipeline全流程的使用说明见[模型开发使用文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001727011)