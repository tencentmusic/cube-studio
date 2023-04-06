[TOC]  
  
## 模型说明  
深度模型主要通过学习到的低维度稠密向量实现模型的泛化能力，从而实现对未见过的内容进行泛化推荐。 很多时候我们不仅需要模型具备泛化能力，同时也希望模型能够通过线性逻辑对模型泛化的规则进行修正。MMoE V2模型在MMoE模型的基础上，提供了多种版本的线性逻辑的引入方式，其设置能够通过参数**use_wide**，**wide_type**，**feature_cross**和**output_cross**的组合来实现，其具体使用方式可参考**参数说明**一节。  
  
## 模板配置  
MMoE V2算法模板配置与MMoE算法模板的配置类似，其示例如下：  
- 【注意】使用者可以先阅读[《pipeline各任务配置》文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117)中的**tensorflow模型训练任务配置.runner封装方式**一节了解模型训练任务的一般配置方法；  
- 【注意】示例中并没有使用到全部可选的模型参数，使用者可根据自己的需求查看**参数说明**一节自行进行配置。  
  
```  
{  
    "num_workers": 1,  
    "node_affin": "only_gpu",  
    "pod_affin": "spread",  
    "timeout": "100d",  
    "job_detail": {  
        "model_input_config_file": "`${PACK_PATH}$`/model_input_config.json",  
        "model_args": {  
            "name": "mmoe_model_v2",  
            "task_structs": [  
                [  
                    64,  
                    16,  
                    1  
                ],  
                [  
                    64,  
                    16,  
                    1  
                ],  
                [  
                    64,  
                    16,  
                    1  
                ]  
            ],  
            "task_use_bias": true,  
            "task_hidden_act": "relu",  
            "task_output_act": "sigmoid",  
            "task_use_bn": false,  
            "task_l2_reg": 0.001,  
            "num_experts": 4,  
            "expert_layers": [  
                128,  
                64  
            ],  
            "expert_use_bias": true,  
            "expert_l2_reg": 0.001,  
            "gate_use_bias": true,  
            "use_wide": true,  
            "wide_type": "FM",  
            "wide_width": 8,  
            "feature_cross": 2,  
            "output_cross": 0,  
            "wide_l2_reg": 0.001  
        },  
        "train_data_args": {  
            "data_file": "${PACK_PATH}$/dataset/train_data.csv",  
            "field_delim": ",",  
            "with_headers": true  
        },  
        "val_data_args": {  
            "data_file": "${PACK_PATH}$/dataset/vali_data.csv",  
            "field_delim": ",",  
            "with_headers": true  
        },  
        "test_data_args": {  
            "data_file": "${PACK_PATH}$/dataset/test_data.csv",  
            "field_delim": ",",  
            "with_headers": true  
        },  
        "train_args": {  
            "batch_size": 256,  
            "epochs": 3,  
            "num_samples": 2938797,  
            "num_val_samples": 1567359,  
            "optimizer": {  
                "type": "adam",  
                "args": {  
                    "learning_rate": 0.001  
                }  
            },  
            "losses": "bce",  
            "metrics": {  
                "is_click": [  
                    "auc",  
                    "bacc"  
                ],  
                "is_favor": [  
                    "auc",  
                    "bacc"  
                ],  
							"play_num1": [  
                    "auc",  
                    "bacc"  
                ]  
            },  
            "train_speed_logger": {  
                "every_batches": 1000,  
                "with_metrics": true  
            },  
            "tensorboard":{  
                "profile_batch": [200,300]  
            },  
            "early_stopping": {  
                "monitor": "val_loss",  
                "mode": "min",  
                "patience": 3,  
                "min_delta": 0.001  
            }  
        },  
        "eval_args": {  
            "batch_size": 256,  
            "metrics": {  
                "is_click": [  
                    "auc",  
                    "bacc"  
                ],  
                "is_favor": [  
                    "auc",  
                    "bacc"  
                ],  
							"play_num1": [  
                    "auc",  
                    "bacc"  
                ]  
            },  
            "output_file": "$EXPORT_PATH$/mmoev2-fm-2-res.json"  
        }  
    }  
}  
  
```  
## 配置参数说明  
在上一节的算法模板配置示例中，我们需要重点关注的是**model_input_config_file**和**model_args**中的内容.  
  
### 输入数据配置  
关于输入数据的配置，即**model_input_config_file**的内容参考 [算法模板数据配置说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001865665)  
- 【注意】MMoE V2模型包括了wide侧和mmoe侧两部分模型，因此在输入数据的配置中尽量使用version 2的配置方式，将wide侧特征和mmoe侧特征分组，并分别使用`wide`和`mmoe`进行命名，一个可供参考的示例如下：  
```  
{  
        "version": 2,  
        "inputs": {  
            "defs":[  
                {"name": "age", "dtype": "int32", "vocab_size":33, "embedding_dim":80},  
                {"name": "age_level", "dtype": "int32", "vocab_size":10, "embedding_dim":80},  
                {"name": "sex", "dtype": "int32", "vocab_size":4, "embedding_dim":80},  
                {"name": "degree", "dtype": "int32", "vocab_size":9, "embedding_dim":80},  
                {"name": "city_level", "dtype": "int32", "vocab_size":9, "embedding_dim":80},  
                {"name": "os_type", "dtype": "int32", "vocab_size":10, "embedding_dim":80},  
                {"name": "genre_0", "dtype": "int32", "vocab_size":61, "embedding_dim":80},  
                {"name": "bd_play_cnt", "dtype": "int32", "vocab_size":59, "embedding_dim":8},  
                {"name": "bd_play_time", "dtype": "int32", "vocab_size":59, "embedding_dim":8},  
                {"name": "wifi_play_cnt", "dtype": "int32", "vocab_size":51, "embedding_dim":8},  
                {"name": "wifi_play_time", "dtype": "int32", "vocab_size":51, "embedding_dim":8},  
                {"name": "is_click", "dtype": "int32", "is_label": true},  
                {"name": "play_num1", "dtype": "int32", "is_label": true},  
                {"name": "play_num5", "dtype": "int32", "is_label": true},  
                {"name": "playtime_3000", "dtype": "int32", "is_label": true},  
                {"name": "is_favor", "dtype": "int32", "is_label": true}  
            ],  
            "groups":{  
                "wide": ["age","age_level","sex","degree","city_level","os_type","genre_0"],  
                "mmoe": ["age","age_level","sex","bd_play_cnt","bd_play_time","wifi_play_cnt"],  
                "labels": ["is_click", "play_num1", "play_num5", "playtime_3000", "is_favor"]  
            }  
        }  
    }  
```  
  
### 模型参数说明  
关于MMoE V2模型的配置，即**model_args**中的内容，具体含义如下：  
- **task_structs**：!!#e06666 必填!!。task tower的网络结构数组，数组中每一个元素是一个task的结构，即task dnn每一层的宽度，从下至上。（!!#ea9999 这里需要注意task的顺序与样本中label的顺序的对应!!）  
- **use_wide**：!!#e06666 非必填!!。是否使用wide侧的网络和特征来作为线性逻辑的修正和补充，True表示使用，False表示不适用，默认为True。  
- **wide_type**：!!#e06666 非必填!!。wide侧网络的类型，可选的类型包括 "FM" 和 "LR"，如果设置为其他值则表示wide侧不使用模型，而是直接使用wide侧特征。!!#ea9999 需要注意该参数的生效要求参数**use_wide**设置为True。  
- **wide_width**：!!#e06666 非必填!!。wide侧模型设置为 "FM"时，该参数生效，表示FM模型中隐向量维度大小。  
- **output_cross**：!!#e06666 非必填!!。表示wide侧模型输出是否mmoe侧模型输出进行融合。设置为0时，wide侧模型输出将于mmoe模型输出相加并用于得到最终的模型整体输出；设置为其他值时，表示wide侧模型输出不直接用于最终结果的计算。  
- **feature_cross**：!!#e06666 非必填!!。wide侧的特征或模型输出与mmoe侧模型的结合方式。  
	- 设置为0时，表示wide侧模型和mmoe侧模型各自使用自己的特征，当**output_cross**设置为0时，wide侧模型输出将于mmoe侧模型输出进行融合；  
	- 设置为1时，表示wide侧的特征会与mmoe侧的特征进行拼接，一同输入到mmoe侧模型的expert和gate中；  
	- 设置为2时，表示wide侧的特征会与mmoe侧的特征进行拼接，一同输入到mmoe侧模型的expert和gate中，同时**wide侧的特征**会与mmoe侧的expert与gate输出的不同task对应的向量进行拼接，再输入到各个task tower中；  
	- 设置为3时，表示wide侧的特征会与mmoe侧的特征进行拼接，一同输入到mmoe侧模型的expert和gate中，同时**wide侧的模型输出**会与mmoe侧的expert与gate输出的不同task对应的向量进行拼接，再输入到各个task tower中；  
	- 设置为4时，表示mmoe侧模型在expert和gate部分将只使用mmoe侧的特征，而同时**wide侧的特征**会与mmoe侧的expert与gate输出的不同task对应的向量进行拼接，再输入到各个task tower中；  
	- 设置为5时，表示mmoe侧模型在expert和gate部分将只使用mmoe侧的特征，而同时**wide侧的模型输出**会与mmoe侧的expert与gate输出的不同task对应的向量进行拼接，再输入到各个task tower中；  
- **wide_l1_reg**：!!#e06666 非必填!!。wide侧网络的参数L1正则化系数，0或者None表示不需要正则化，默认为None。  
- **wide_l2_reg**：!!#e06666 非必填!!。wide侧网络的参数L2正则化系数，0或者None表示不需要正则化，默认为None。  
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
- **name**：!!#e06666 非必填!!。模型名字，可以自定义。默认为"mmoe v2"。