YoutubeDNN算法模板配置示例如下（可以先阅读[《pipeline各任务配置》文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117)中的"tensorflow模型训练任务配置.runner封装方式"一节了解模型训练任务的一般配置方法）:  
  
```   
{  
    "num_workers": 10,  
    "node_affin": "only_cpu",  
    "pod_affin": "spread",  
    "timeout": "100d",  
    "resources": {  
        "limits": {  
            "cpu": "10",  
            "memory": "16G"  
        }  
    },  
    "job_detail": {  
        "model_input_config_file": "${PACK_PATH}$/model_input_mini.json",  
        "user_embedding_file": "${DATA_PATH}$/user_embeddings.csv",  
        "item_embedding_file": "${DATA_PATH}$/item_embeddings.txt",  
        "predict_data_file": "${DATA_PATH}$/../20210309-213030.017942/user_hist_trans-part-*.csv",  
        "item_embedding_predict_parallel": 1,  
        "user_embedding_predict_parallel": 20,  
        "item_embedding_fmt": "ef",  
		      "user_embedding_fmt": "",  
        "predict_batch_size": 5000,  
        "uid_col_name": "uin",  
        "model_args": {  
            "dnn_hidden_layers": [  
                256,  
                128  
            ],  
            "dnn_hidden_act_fn": "relu",  
            "dnn_output_act_fn": "lrelu",  
            "name": "youtubednn"  
        },  
        "train_data_args": {  
            "data_file": "${DATA_PATH}$/../20210309-213030.017942/train_data_trans-part-*.csv",  
            "field_delim": " ",  
            "file_type": "csv",  
            "shard_policy": "FILE"  
        },  
        "train_args": {  
            "mw_com": "RING",  
            "train_type": "compile_fit",  
            "batch_size": 512,  
            "epochs": 1,  
            "num_samples": 20000000,  
            "validation_steps": 1,  
            "optimizer": {  
                "type": "adam",  
                "args": {  
                    "learning_rate": 0.005  
                }  
            },  
            "losses": {  
                "type": "sampled_softmax_loss",  
                "args": {  
                    "num_samples": 20  
                }  
            },  
            "metrics": [  
                [  
                    {  
                        "type": "topk_hitrate",  
                        "args": {  
                            "k": 50,  
                            "name": "top50_hitrate"  
                        }  
                    },  
                    {  
                        "type": "topk_hitrate",  
                        "args": {  
                            "k": 100,  
                            "name": "top100_hitrate"  
                        }  
                    },  
                    {  
                        "type": "topk_hitrate",  
                        "args": {  
                            "k": 200,  
                            "name": "top200_hitrate"  
                        }  
                    },  
                    {  
                        "type": "topk_hitrate",  
                        "args": {  
                            "k": 500,  
                            "name": "top500_hitrate"  
                        }  
                    }  
                ]  
            ],  
            "early_stopping": {  
                "monitor": "top50_hitrate",  
                "mode": "max",  
                "patience": 3,  
                "min_delta": 0.001  
            },  
            "train_speed_logger": {  
                "with_metrics": true  
            }  
        },  
        "eval_args": {  
            "batch_size": 1024,  
            "metrics": [  
                [  
                    {  
                        "type": "topk_hitrate",  
                        "args": {  
                            "k": 50,  
                            "name": "top50_hitrate"  
                        }  
                    },  
                    {  
                        "type": "topk_hitrate",  
                        "args": {  
                            "k": 100,  
                            "name": "top100_hitrate"  
                        }  
                    },  
                    {  
                        "type": "topk_hitrate",  
                        "args": {  
                            "k": 200,  
                            "name": "top200_hitrate"  
                        }  
                    },  
                    {  
                        "type": "topk_hitrate",  
                        "args": {  
                            "k": 500,  
                            "name": "top500_hitrate"  
                        }  
                    }  
                ]  
            ]  
        }  
    }  
}  
```  
  
上面配置中，主要关注**model_input_config_file**和**model_args**中的内容.  
关于输入数据的配置见[算法模板数据配置说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001865665)  
**model_input_config_file**：指定了模型输入描述文件，输入描述文件的说明见[算法模板输入描述文件格式说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001851927)  
**load_model_from**：可以指定模型加载路径，如果配置了这个参数，且路径真是存在，则会跳过训练过程，从路径加载模型。这个可以实现两个目的：1.用于不想训练，只是想用评估数据跑下之前训好模型的指标的场景（配置了val_data_args和eval_args）2.使用已经训练好的模型预测生成user embedding文件和item embedding文件  
**user_embedding_file**：结果user embedding文件存放路径，如果为空，则不作user embedding预测。  
**item_embedding_file**：结果item embedding文件存放路径，如果为空，则不作item embedding预测。（!!#e06666  !17 item embedding导出会使用训练数据作为预测数据源，**训练数据格式必须是csv**!!!）  
**predict_data_file**：用于预测**!!#cc0000 user embedding!!**的数据文件，如果为空则不作user embedding预测。!!#e06666  !17 **文件格式目前必须为csv**，且必须包含文件头，且列必须用空格分隔! !!。uid列名需通过 *uid_col_name* 指定。（!!#e06666 !17  注意predict_data_file除了uid列外，其他列必须至少包含与训练数据同样的列! !!）  
**uid_col_name**：predict_data_file文件中uid对应的列名。  
**predict_batch_size**：预测embedding时的batch大小。  
**item_embedding_predict_parallel**：预测item embedding时的并行度。默认为1。速度和内存占用都会线性增加。  
**user_embedding_predict_parallel**：预测user embedding时的并行度。默认为1。速度和内存占用都会线性增加。  
**item_embedding_fmt**：生成item embedding文件的格式，有三种取值：  
 - **ef**：表示生成EF的向量格式，每一行是格式为 *<item_id>|<item_id>|<embedding_string>* ，其中 *<embedding_string>* 是逗号分割的各维度。  
	- **ts:<BN>**：表示生成terasearch的向量格式，具体为每一行 *{"MD": "<MD>", "BN": "<BN>", "VA": "txt|<embedding_string>"}* ,其中 *<embedding_string>* 是逗号分割的各维度  
 - 空字符串：表示默认格式，具体为每一行 *<item_it>	<embedding_string>* ，其中 *<embedding_string>* 是逗号分割的各维度。  
  
**user_embedding_fmt**：生成user embedding文件的格式。取值方式与**item_embedding_fmt**一致  
**model_args**中是特定于YoutubeDNN模型的参数：  
- **dnn_hidden_layers**：!!#e06666 必填!!。dnn的隐层结构，即每层的宽度大小（不包括输入和输出层），从下至上。  
- **dnn_hidden_act_fn**：!!#e06666 非必填!!。dnn隐藏层的激活函数，!!#e06666 如果设置为一个字符串，例如"relu"，则表示所有隐层的激活函数都是一样的；如果需要每层指定不同的激活函数，则可以设置为数组，例如["relu", "sigmoid"]，表示第一层使用relu，第二层使用sigmoid，层顺序与  *dnn_hidden_layers*  一致。设置的数组长度不一定与隐层数量一样，当  *dnn_hidden_act_fn* 长度超过隐层数量时，超出的部分会被自动忽略，如果比隐层数量短，则缺少部分对应的隐层就没有激活函数。如果是某个中间位置隐层不需要激活函数，而上下都需要，则对应位置设置为"none"即可，例如有三个隐层，其中第一和第三层分别是relu和sigmoid，而第二层不需要，则可以设置为["relu", "none", "sigmoid"]。!!  
- **dnn_output_act_fn**：!!#e06666 非必填!!。dnn输出层的激活函数，默认为"relu"。  
- **dnn_use_bn**：!!#e06666 非必填!!。dnn是否使用batch normalization，!!#e06666 如果设置成一个值，例如true，则表示所有层都使用batch normalize；如果各层不一样，则可以设置成数组，例如[true, false]，表示第一层使用BN，第二层不使用。与 *dnn_hidden_act_fn* 类似，数组长度也不一定与隐层数量一样，长度不一样时，对应关系与 *dnn_hidden_act_fn* 中说明一样。!!  
- **dnn_dropout**：!!#e06666 非必填!!。dnn的层间dropout概率，!!#e06666 如果设置成一个值，例如0.5，则表示所有隐层都采用0.5的dropout；如果需要每层采用不同的dropout值，则可以设置为数组，例如[0.5, 0.2]，表示第一层采用0.5的dropout，第二层采用0.2的dropout。与 *dnn_hidden_act_fn* 类似，数组长度也不一定与隐层数量一样，长度不一样时，对应关系与 *dnn_hidden_act_fn* 中说明一样。!!  
- **dnn_l1_reg**：!!#e06666 非必填!!。task网络的参数L1正则化系数，0或者None表示不需要正则化，默认为None。  
- **dnn_l2_reg**：!!#e06666 非必填!!。task网络的参数L2正则化系数，0或者None表示不需要正则化，默认为None。  
- **name**：!!#e06666 非必填!!。模型名字，可以自定义。默认为"youtubednn"。  
  
**!!#e06666 注意，YoutubeDNN的loss和metric类型只能是sampled softmax和topk hitrate。!!**  
sampled softmax的参数如下：  
- **num_samples**：计算loss时，对于每个正样本，负样本采样的个数。  
- **sample_algo**：采样策略，支持如下几种采样策略，默认为"learned"：  
	- "**uniform**"：均匀采样，及所有类别被采样到的概率是一样的  
    - "**log_uniform**"：假设类别分布为Zipfian，即每个类被采样到的概率为p(class) = (log(class+2)-log(class+1))/log(max+1)，class是类编号，max是最大编号，!!#ff0000 使用这个策略时需要实现对各个类别按照频率从大到小进行编号，即频率越大的类编号应该越小!!  
	- "**learned**"：在训练过程中对类别计数统计，初始时所有类别是均匀的，根据训练过程看到的数据逐步调整类别分布。  
	- "**fixed**"：如果用户实现知道类别分布，可以使用此策略传入类别分布信息，传递方式见下面sample_algo_params参数说明。  
- **sample_algo_params**：采样策略参数，!!#ff0000 **只对fixed策略有意义**，!!参数形式是一个dict，支持的字段有：  
	- **vocab_file**：以文件方式指定类别的分布，每一行对应一个类别的分布权重，这个权重可以是类的频数也可以是类的概率值。  
	- **unigrams**：以数组方式指定类别分布，数组中每一项对应一个类别的分布权重，这个权重可以是类的频数也可以是类的概率值。unigrams和vocab_file必须至少指定一个，如果同时指定，则以unigrams为准。  
	- **distortion**：!!#ff0000 非必填!!，对类别分布的扰动值，即每个类别的分布权重会先变成自己的distortion次方，因此distortion为1时跟原始分布一样，为0时就变成均匀分布。默认为1  
	- **num_reserved_ids**：!!#ff0000 非必填!!，类别编号的起始值，默认为0。  
  
topk hitrate metric的参数如下：  
- **k**：top几  
- **name**：metric名字  
其他参数的意义都可以参考《pipeline各任务配置》文档。  
  
pipeline全流程的使用说明见[模型开发使用文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001727011)