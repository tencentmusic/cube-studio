[TOC]  
  
## 模型说明  
ComiRec模型是阿里巴巴团队在 KDD2020 的论文 Controllable Multi-Interest Framework for Recommendation 中提出，这是一个用于推荐召回阶段的序列化模型。召回阶段需要根据用户的兴趣从海量的商品中去检索出相关候选 Item，满足推荐相关性和多样性需求，与MIND模型类似，ComiRec同样关注用户的多兴趣召回。  
  
如下图所示，如果将推荐表述成一个序列化问题，需要通过用户的历史行为序列，估计用户此后会感兴趣的物品。而在现实应用中，用户的历史序列很可能是发散的，即序列中有许多不同品类的物品，表现出用户具有多个不同的兴趣，如果对用户行为使用一个统一的融合向量来进行表征，那么这个向量要完整地表达出用户的多个兴趣则存在较大困难。  
![user-mutli-interests](ComiRec1.png)  
  
ComiRec模型采用了两种方法来建模用户的多兴趣，分别是动态路由算法 Dynamic Routing 和自注意力机制 Self-Attention。其中动态路由算法与MIND模型使用的方法类似，仅在参数初始化和兴趣数量选择上有所区别，但是其表达用户兴趣并没有像MIND模型一样在经过胶囊网络后再经过两层DNN，而是将胶囊网络的结果作为用户多兴趣表征。需要注意的是，ComireRec使用的所谓Self-Attention机制并不是 《Attention is All You Need》中提到的Transformer模型中的Self-Attention机制，其并未采用$softmax(\frac{QK^T}{\sqrt(d)})$的方式来计算注意力分数，而是使用DNN来替代$QK^T$点积运算。ComiRec模型并不支持引用额外的用户属性，其用户兴趣表征利用的是Item ID的历史序列，具体模型结构如下图所示：  
![comirec](ComiRec2.png)  
  
## 模板配置  
ComiRec算法模板的配置示例如下：  
- 【注意】使用者可以先阅读[《pipeline各任务配置》文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117)中的**tensorflow模型训练任务配置.runner封装方式**一节了解模型训练任务的一般配置方法；  
- 【注意】示例中并没有使用到全部可选的模型参数，使用者可根据自己的需求查看**参数说明**一节自行进行配置。  
```  
{  
    "num_workers": 1,  
    "node_affin": "only_cpu",  
    "pod_affin": "spread",  
    "timeout": "100d",  
    "job_detail": {  
        "model_input_config_file": "${PACK_PATH}$/input_config_data.json",  
        "model_args": {  
            "name": "comirec_model",  
            "item_embedding_dim": 64,   
            "seq_max_len": 50,   
            "interest_extractor": "DR",  
            "num_interests": 3,   
            "add_pos": true,  
            "pow_p": 1,   
            "hidden_size": null  
        },  
        "train_data_args": {  
            "data_file": "${PACK_PATH}$/train_data.csv",  
            "field_delim": ",",  
            "with_headers": true  
        },  
        "val_data_args": {  
            "data_file": "${PACK_PATH}$/test_data.csv",  
            "field_delim": ",",  
            "with_headers": true  
        },  
        "test_data_args": {  
            "data_file": "${PACK_PATH}$/test_data.csv",  
            "field_delim": ",",  
            "with_headers": true  
        },  
		"predict_data_args": {  
            "data_file": "${PACK_PATH}$/predict_data.csv",  
            "field_delim": ",",  
            "with_headers": true,  
            "model_input_config_file": "${PACK_PATH}$/item_config.json"  
        },  
        "train_args": {  
            "batch_size": 512,  
            "epochs": 20,  
            "train_type": "compile_fit",  
            "num_samples": 100000,  
            "num_val_samples": 50000,  
            "optimizer": {  
                "type": "adam",  
                "args": {  
                    "learning_rate": 0.001  
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
            "train_speed_logger": {  
                "every_batches": 1000,  
                "with_metrics": true  
            },  
            "tensorboard":{  
                "profile_batch": [200,300]  
            },  
            "early_stopping": {  
                "monitor": "top50_hitrate",  
                "mode": "max",  
                "patience": 3,  
                "min_delta": 0.001  
            }  
        },  
        "eval_args": {  
            "batch_size": 512,  
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
            "output_file": "${DATA_PATH}$/comirec-res.json"  
        },  
		"predict_args": {  
            "model_path": "${DATA_PATH}$/saved_model/comirec_model-item_embeddings_model/",  
            "batch_size": 1024,  
            "result_field_delim": " ",  
            "output_delim": ",",  
            "write_headers": true  
        }  
    }  
}  
  
```  
## 配置参数说明  
在上一节的算法模板配置示例中，我们需要重点关注的是**导出embedding的配置**，以及**model_input_config_file**和**model_args**中的内容。  
  
### 导出embedding配置  
#### 新导出方式配置  
目前ComiRec模型支持通过配置predict_args的方式来进行item embedding的预测，具体配置参数及含义可参考 [pipeline各任务配置说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117) 中的 [**tensorflow模型离线预测任务配置**](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc13) 小节。  
  
以本算法模板中使用的配置为例：  
```  
"predict_data_args": {  
    "data_file": "${PACK_PATH}$/predict_data.csv",  
    "field_delim": ",",  
    "with_headers": true,  
    "model_input_config_file": "${PACK_PATH}$/item_config.json"  
},  
...  
"predict_args": {  
    "model_path": "${DATA_PATH}$/saved_model/comirec_model-item_embeddings_model/",  
    "batch_size": 1024,  
    "result_field_delim": " ",  
    "output_delim": ",",  
    "write_headers": true  
}  
```  
通过`predict_data_args`来配置需要预测的item的数据文件和描述文件，数据文件通过`data_file`配置，其中包含与训练过程中相同的item特征列，描述文件通过`model_input_config_file`配置，其中各特征的`embedding_name, embedding_dim`等设置应该与训练过程中的数据描述文件相同。  
通过`predict_args`来配置预测结果文件的生成，其中`model_path`需要配置为本次任务中保存下来的 Item_Embedding 模型，其位置在`${DATA_PATH}$/saved_model/comirec_model-item_embeddings_model/`下，其他设置可参考 [**tensorflow模型离线预测任务配置**](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117@toc13) 。  
  
#### 旧导出方式配置  
旧导出方式与YouTubeDNN导出item embedding相似，如果使用旧的导出方式，需要在job_detail中进行配置，本小节提供了一个简单的配置示例，具体参数含义如下：  
```  
"job_detail": {  
        "model_input_config_file": "${PACK_PATH}$/input_config_data.json",  
        "user_embedding_file": "${DATA_PATH}$/user_embeddings.csv",  
        "item_embedding_file": "${DATA_PATH}$/item_embeddings.txt",  
        "predict_data_file": "${PACK_PATH}$/predict_data.csv",  
        "item_embedding_predict_parallel": 1,  
        "user_embedding_predict_parallel": 20,  
        "predict_batch_size": 1000,  
        "uid_col_name": "uin",  
		...  
```  
- **load_model_from**：可指定模型加载路径，如果配置了该参数且路径真实存在，则会跳过训练过程，从路径加载模型。该参数实现两个目的：  
	- 不训练模型，只用于使用数据来评估已经训练好的模型的指标（需要配置val_data_args和eval_args）；  
	- 使用已经训练好的模型预测生成user embedding文件和item embedding文件。  
- **user_embedding_file**：生成user embedding时结果文件的存放路径，如果为空，则不进行user embedding预测。需要注意的是，ComiRec模型默认生成的是user的多个兴趣，生成的user embedding文件有k+1列，列之间用制表符分隔，第一列为uid，后续每一列为用户兴趣的embedding表示，embedding各维度用逗号分隔。  
- **item_embedding_file**：生成item embedding时结果文件的存放路径，如果为空，则不进行item embedding预测。需要注意的是，目前ComiRec模型在导出item embedding时只支持导出item id的embedding结果，生成的item embeding中每一行的格式为 !!#e06666 `{"MD": "<id>","BN": "push_playlist_recall", "VA": "txt|<embedding>"} `!!，其中`<id>`表示item id，`<embedding>`表示对应embedding，embedding各维度用逗号分隔。  
- **predict_data_file**：用于预测item/user embedding的数据文件，如果为空，则item和user embedding预测都不会进行。!!#e06666 文件格式目前必须为csv，且必须包含文件头!!。item id列对应的是model_input_config_file中指定为label的那个列。uid列名需通过uid_col_name指定。（!!#e06666 注意predict_data_file除了uid列外，其他列必须至少包含与训练数据同样的列!!）  
- **uid_col_name**：predict_data_file文件中uid对应的列名。  
- **predict_batch_size**：预测embedding时的batch大小。  
- **item_embedding_predict_parallel**：预测item embedding时的并行度。默认为1。速度和内存占用都会线性增加。  
- **user_embedding_predict_parallel**：预测user embedding时的并行度。默认为1。速度和内存占用都会线性增加。  
  
### 输入数据配置  
关于输入数据的配置，即**model_input_config_file**的内容参考 [算法模板数据配置说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001865665)，一个可供参考的ComiRec模型输入数据配置如下：  
- 【注意】ComiRec模型的输入只需要包括用户历史行为特征（特征组使用`item`命名），且根据论文的描述应该只含有表示Item ID的一项特征；此外如果需要训练ComiRec模型，还需要包括训练时用于计算Attention的的目标Item的特征（特征组使用`target`命名）；  
- 【注意】ComiRec模型的输入中，`item`特征组是必需的，而模型在训练时，`target`特征组也是必需的；  
- 【注意】ComiRec模型的输入中，`item`特征组内的特征需要是序列特征，序列用分隔符`val_sep`串联起来表示，每个特征需要将`embedding_combiner`设置为`seq_pad`，并且可以指定将序列padding到的最大长度`max_len`（不指定则会padding到128）；`target`特征组内的特征由于不是序列特征而是单个item，因此`embedding_combiner`不需要设置为`seq_pad`；  
- 【注意】ComiRec模型的输入中，`item`特征组、`target`特征组和`labels`特征**必须要指定`embedding_name`**，并且对于含义相同的特征，`embedding_name`**必须相同**，相应的`vocab_size`和`embedding_dim`等也需要保持一致；以参考示例作为说明，`item`特征组中的"list_item_id"和`target`特征组中的"target_item_id"以及`labels`中的"item_id"都是表示物品Item的ID，因此其`embedding_name`都保持一致，都为"item_id"；  
- 【注意】ComiRec模型的输入配置中，应该将`item`特征组的特征说明放置在`target`和`labels`特征说明的前面；  
- 【注意】ComiRec模型的输入中，如果有每个序列特征的真实长度特征，应将其命名为`hist_len`，可将其分类到`item_len`特征组  
```  
{  
    "version": 2,  
    "inputs": {  
        "defs":[  
            {"name": "list_item_id", "dtype": "int32", "val_sep":",", "vocab_size":2000, "embedding_dim":32, "max_len":20, "embedding_combiner": "seq_pad", "embedding_name":"item_id"},  
            {"name": "target_item_id", "dtype": "int32", "embedding_name":"item_id", "vocab_size":2000, "embedding_dim":32},  
            {"name": "item_id", "dtype": "int32", "embedding_name":"item_id", "vocab_size":2000, "embedding_dim":32, "is_label":true}  
        ],  
        "groups":{  
            "item":["list_item_id"],  
            "target":["target_item_id"]  
            "labels":["item_id"]  
        }  
    }  
}  
```  
  
### 模型参数说明  
关于ComiRec模型的配置，即**model_args**中的内容，具体含义如下：  
  
- **item_embedding_dim**：!!#e06666 必填!!。item embedding的维度，需要和输入描述文件中的表示item的特征使用的`embedding_dim`保持一致。  
- **seq_max_len**：!!#e06666 必填!!。序列特征的最大长度，需要和输入描述文件中`item`特征组中的各序列特征使用的`max_len`保持一致，如果特征中未设置最大长度，则需要设置该值为128。  
- **num_interests**：!!#e06666 非必填!!。用户兴趣数量，ComiRec模型中的用户多兴趣提取层（胶囊网络或注意力网络）会根据该值来产生多个不同的用户兴趣表征，默认为3。  
- **interest_extractor**：!!#e06666 非必填!!。提取用户多兴趣的方式，可选择`DR`或者`SA`，分别表示使用动态路由算法和自注意力机制，默认为`DR`。  
- **hidden_size**：!!#e06666 非必填!!。使用自注意力机制作为用户多兴趣提取时的DNN第一层的输出维度。如果不设置，则默认为`item_embedding_dim`\*4。  
- **pow_p**：!!#e06666 非必填!!。调整模型最后用户兴趣与目标Item表征的注意力计算结果中注意力分布的可调参数，默认为1。当p接近0时，每个兴趣将得到均匀的注意力，当p大于1时，随着p的增加，注意力计算中点积越大的值将得到越多的注意力。  
- **name**：!!#e06666 非必填!!。模型名字，可以自定义。默认为"ComiRec"。  
  
**!!#e06666 注意，目前ComiRec的loss和metric类型只能是sampled softmax和topk hitrate。!!**  
  
其中sampled softmax的参数如下：  
- **num_samples**：计算loss时，对于每个正样本，负样本采样的个数。  
- **sample_algo**：采样策略，支持如下几种采样策略，默认为"learned"：  
	- "**uniform**"：均匀采样，及所有类别被采样到的概率是一样的  
    - "**log_uniform**"：假设类别分布为Zipfian，即每个类被采样到的概率为$ p(class) = (log(class+2)-log(class+1))/log(max+1) $，class是类编号，max是最大编号，!!#ff0000 使用这个策略时需要实现对各个类别按照频率从大到小进行编号，即频率越大的类编号应该越小!!  
	- "**learned**"：在训练过程中对类别计数统计，初始时所有类别是均匀的，根据训练过程看到的数据逐步调整类别分布。  
	- "**fixed**"：如果用户实现知道类别分布，可以使用此策略传入类别分布信息，传递方式见下面sample_algo_params参数说明。  
- **sample_algo_params**：采样策略参数，!!#ff0000 **只对fixed策略有意义**，!!参数形式是一个dict，支持的字段有：  
	- **vocab_file**：以文件方式指定类别的分布，每一行对应一个类别的分布权重，这个权重可以是类的频数也可以是类的概率值。  
	- **unigrams**：以数组方式指定类别分布，数组中每一项对应一个类别的分布权重，这个权重可以是类的频数也可以是类的概率值。unigrams和vocab_file必须至少指定一个，如果同时指定，则以unigrams为准。  
	- **distortion**：!!#ff0000 非必填!!，对类别分布的扰动值，即每个类别的分布权重会先变成自己的distortion次方，因此distortion为1时跟原始分布一样，为0时就变成均匀分布。默认为1  
	- **num_reserved_ids**：!!#ff0000 非必填!!，类别编号的起始值，默认为0。  
  
其中topk hitrate metric的参数如下：  
- **k**：表示 top k  
- **name**：metric名字  
  
其他参数的意义都可以参考《pipeline各任务配置》文档。  
pipeline全流程的使用说明见[模型开发使用文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001727011)  
  
## SERVING阶段User Multi-Interest Model的使用  
与MIND模型类似，ComiRec模型在训练完成并保存模型时，会保存两个模型，其中一个是用于生成item embedding的Item模型，另一个是用于在线上动态更新用户多兴趣的User Multi-Interest模型。  
User Multi-Interest模型本质上就是一个完整的ComiRec模型，在Training阶段，由于输入数据中存在target item的特征，因此模型会与target item进行Label Aware Attention的计算用于模型训练；在Serving阶段，加载位于`${DATA_PATH}$/saved_model/comirec_model/`下的ComiRec模型，在输入特征中不包含训练阶段的target item特征（需要注意，模型输入特征除target item特征之外应该与训练阶段的输入相同），则模型将会输出用户的多兴趣。