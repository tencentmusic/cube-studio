ESMM论文：https://arxiv.org/pdf/1804.07931.pdf  
ESMM算法模板配置示例如下（可以先阅读[《pipeline各任务配置》文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001722117)中的"tensorflow模型训练任务配置.runner封装方式"一节了解模型训练任务的一般配置方法）:  
  
```   
{  
    "num_workers": 1,  
    "node_affin": "only_cpu",  
    "job_detail": {  
        "model_input_config_file": "${PACK_PATH}$/model_input_demo.json",  
        "load_model_from": "${DATA_PATH}$/saved_model/esmm",  
        "model_args": {  
            "cvr_dnn_units": [  
                64,  
                32,  
                1  
            ],  
            "cvr_dnn_hidden_act": "lrelu",  
            "cvr_dnn_use_bias": true,  
            "cvr_dnn_use_bn": false,  
            "cvr_dnn_dropout": 0,  
            "cvr_dnn_l1_reg": 0.003,  
            "cvr_dnn_l2_reg": 0.005,  
            "ctr_dnn_units": [  
                64,  
                32,  
                1  
            ],  
            "ctr_dnn_hidden_act": "lrelu",  
            "ctr_dnn_use_bias": true,  
            "ctr_dnn_use_bn": false,  
            "ctr_dnn_dropout": 0,  
            "ctr_dnn_l1_reg": 0.003,  
            "ctr_dnn_l2_reg": 0.005,  
            "ctr_label_name": "is_kg",  
            "ctcvr_label_name": "is_dapan"  
        },  
        "train_data_args": {  
            "data_file": "${PACK_PATH}$/part-00099",  
            "file_type": "csv",  
            "field_delim": ",",  
            "with_headers": false,  
            "headers": "neg_sample,logid,moment_id,is_click,age_level,sex,degree,income_level,city_level,is_kg,log_days,net_day,is_dapan,os_type,square_clicknum_1d,square_clicknum_3d,square_clicknum_7d,square_clicknum_15d,square_fig_clicknum_1d,square_fig_clicknum_3d,square_fig_clicknum_7d,square_fig_clicknum_15d,square_video_clicknum_1d,square_video_clicknum_3d,square_video_clicknum_7d,square_video_clicknum_15d,square_fig_viewtime_1d,square_fig_viewtime_3d,square_fig_viewtime_7d,square_fig_viewtime_15d,square_video_viewtime_1d,square_video_viewtime_3d,square_video_viewtime_7d,square_video_viewtime_15d,total_viewcnt_1d,total_viewcnt_3d,total_viewcnt_7d,total_viewcnt_15d,total_viewtime_1d,total_viewtime_3d,total_viewtime_7d,total_viewtime_15d,total_commentcnt_1d,total_commentcnt_3d,total_commentcnt_7d,total_commentcnt_15d,total_favorcnt_1d,total_favorcnt_3d,total_favorcnt_7d,total_favorcnt_15d,cate_1_7d,cate_1_15d,cate_1_30d,cate_1_90d,cate_2_7d,cate_2_15d,cate_2_30d,cate_2_90d,cate_3_7d,cate_3_15d,cate_3_30d,cate_3_90d,group_90d,publisher_90d,moment_type,pics_num,video_duration,freshness,validity_level,is_artist_self,cate_first,cate_second,cate_third,group_id,publisher_id,moment_type_n,pics_num_n,video_duration_n,freshness_n,validity_level_n,is_artist_self_n,cate_first_n,cate_second_n,cate_third_n,group_id_n,publisher_id_n"  
        },  
        "val_data_args": {  
            "data_file": "${PACK_PATH}$/part-00099",  
            "file_type": "csv",  
            "field_delim": ",",  
            "with_headers": false,  
            "headers": "neg_sample,logid,moment_id,is_click,age_level,sex,degree,income_level,city_level,is_kg,log_days,net_day,is_dapan,os_type,square_clicknum_1d,square_clicknum_3d,square_clicknum_7d,square_clicknum_15d,square_fig_clicknum_1d,square_fig_clicknum_3d,square_fig_clicknum_7d,square_fig_clicknum_15d,square_video_clicknum_1d,square_video_clicknum_3d,square_video_clicknum_7d,square_video_clicknum_15d,square_fig_viewtime_1d,square_fig_viewtime_3d,square_fig_viewtime_7d,square_fig_viewtime_15d,square_video_viewtime_1d,square_video_viewtime_3d,square_video_viewtime_7d,square_video_viewtime_15d,total_viewcnt_1d,total_viewcnt_3d,total_viewcnt_7d,total_viewcnt_15d,total_viewtime_1d,total_viewtime_3d,total_viewtime_7d,total_viewtime_15d,total_commentcnt_1d,total_commentcnt_3d,total_commentcnt_7d,total_commentcnt_15d,total_favorcnt_1d,total_favorcnt_3d,total_favorcnt_7d,total_favorcnt_15d,cate_1_7d,cate_1_15d,cate_1_30d,cate_1_90d,cate_2_7d,cate_2_15d,cate_2_30d,cate_2_90d,cate_3_7d,cate_3_15d,cate_3_30d,cate_3_90d,group_90d,publisher_90d,moment_type,pics_num,video_duration,freshness,validity_level,is_artist_self,cate_first,cate_second,cate_third,group_id,publisher_id,moment_type_n,pics_num_n,video_duration_n,freshness_n,validity_level_n,is_artist_self_n,cate_first_n,cate_second_n,cate_third_n,group_id_n,publisher_id_n"  
        },  
        "train_args": {  
            "mw_com": "RING",  
            "train_type": "compile_fit",  
            "batch_size": 512,  
            "epochs": 1,  
            "optimizer": {  
                "type": "adam",  
                "args": {  
                    "learning_rate": 0.001  
                }  
            },  
            "losses": "bce",  
            "metrics": {  
                "is_kg": [  
                    "auc",  
                    "bacc"  
                ],  
                "is_dapan": [  
                    "auc",  
                    "bacc"  
                ]  
            },  
            "early_stopping": {  
                "monitor": "is_kg_auc",  
                "mode": "max",  
                "patience": 5,  
                "min_delta": 0.001  
            },  
            "train_speed_logger": {  
                "every_batches": 1000,  
                "with_metrics": true  
            },  
            "tensorboard": {}  
        },  
        "eval_args": {  
            "batch_size": 256,  
            "metrics": {  
                "is_kg": [  
                    "auc",  
                    "bacc"  
                ],  
                "is_dapan": [  
                    "auc",  
                    "bacc"  
                ]  
            }  
        }  
    }  
}  
```  
上面配置中，主要关注 **model_input_config_file** 和 **model_args** 中的内容.  
关于输入数据的配置见[算法模板数据配置说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001865665)  
**model_input_config_file** ：指定了模型输入描述文件，输入描述文件的说明见[算法模板输入描述文件格式说明](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001851927)。!!#cc0000 注意输入特征需要分组，用于cvr预估的特征分到 ***cvr*** 组，用于ctr预估的特征分到 ***ctr*** 组。 !!  
**load_model_from** ：可以指定模型加载路径，如果配置了这个参数，且路径真是存在，则会跳过训练过程，从路径加载模型。这个主要是用于不想训练，只是想用评估数据跑下之前训好模型的指标的场景（配置了val_data_args和eval_args）  
**model_args** 中是特定于ESMM模型的参数：  
- **cvr_dnn_units** ：!!#cc0000 必填!!。CVR预估DNN的网络结构，数组中每一个元素是网络对应层的宽度，**从下至上**。**!!#cc0000 网络输出层宽度应该是1，如果大于1，会自动在最后补上宽度1的输出层。!!**  
- **cvr_dnn_hidden_act**：!!#cc0000 非必填!!。CVR预估DNN的隐层（即除最后输出层的其他层）的激活函数类型，如果设置为一个字符串，例如"relu"，则表示所有隐层的激活函数都是一样的；!!#e06666 如果需要每层指定不同的激活函数，则可以设置为数组，例如["relu", "sigmoid"]，表示第一层使用relu，第二层使用sigmoid，层顺序与 user_tower_units 一致。设置的数组长度不一定与隐层数量一样，当 user_tower_hidden_act 长度超过隐层数量时，超出的部分会被自动忽略，如果比隐层数量短，则缺少部分对应的隐层就没有激活函数。如果是某个中间位置隐层不需要激活函数，而上下都需要，则对应位置设置为"none"即可，例如有三个隐层，其中第一和第三层分别是relu和sigmoid，而第二层不需要，则可以设置为["relu", "none", "sigmoid"]。!!  
- **cvr_dnn_use_bias**：!!#cc0000 非必填!!。CVR预估DNN每一层在线性变换时是否加入偏置量。默认为True。  
- **cvr_dnn_use_bn**：!!#cc0000 非必填。!!CVR预估DNN各层（包含输出层）是否使用batch normalization。如果设置成一个值，例如true，则表示所有层都使用batch normalize；如果各层不一样，则可以设置成数组，例如[true, false]，表示第一层使用BN，第二层不使用。与cvr_dnn_hidden_act类似，数组长度也不一定与隐层数量一样，长度不一样时，对应关系与cvr_dnn_hidden_act中说明一样。  
- **cvr_dnn_dropout**：!!#cc0000 非必填。!!CVR预估DNN隐层的dropout的概率，如果设置成一个值，例如0.5，则表示所有隐层都采用0.5的dropout；如果需要每层采用不同的dropout值，则可以设置为数组，例如[0.5, 0.2]，表示第一层采用0.5的dropout，第二层采用0.2的dropout。与cvr_dnn_hidden_act类似，数组长度也不一定与隐层数量一样，长度不一样时，对应关系与cvr_dnn_hidden_act中说明一样。  
- **cvr_dnn_l1_reg**：!!#cc0000 非必填!!。CVR预估DNN的参数L1正则化系数。  
- **cvr_dnn_l2_reg**：!!#cc0000 非必填!!。CVR预估DNN的参数L2正则化系数。  
- **ctr_***：CTR预估DNN相关配置。与CVR对应配置方法一致。  
- **ctr_label_name**：对应ctr的label列名，需要与model_input_config_fle里面的列名对应一致。  
- **ctcvr_label_name**：对应ctcvr的label里面，需要与model_input_config_fle里面的列名对应一致。  
  
其他参数的意义都可以参考《pipeline各任务配置》文档。  
pipeline全流程的使用说明见[模型开发使用文档](http://tapd.oa.com/kubeflow/markdown_wikis/show/#1220424693001727011)  
**ESMM模型训练完成之后，会在模型保存目录下生成三个子目录：esmm，esmm-ctr，esmm-cvr，分别对应完整ESMM模型（输出ctr预测值和ctcvr预测值），ctr预估模型（输出ctr预测值），cvr预估模型（输出cvr预测值）。三个模型都可以单独用作serving，ctr预估模型输入与模型输入描述中ctr特征组部分一致；cvr预估模型输入与cvr特征组一致。ESMM模型的输入则包含了cvr和ctr两部分。**