# xgboost 模板
描述：单机xgb训练，支持训练预测。

镜像：ccr.ccs.tencentyun.com/cube-studio/xgb:20230801 


启动参数：  
```bash
{
    "训练推理": {
      "--train_dataset": {
        "type": "str",
        "item_type": "str",
        "label": "训练数据集",
        "require": 0,
        "choice": [],
        "range": "",
        "default": "",
        "placeholder": "",
        "describe": "训练数据集",
        "editable": 1
      },
      "--val_dataset": {
        "type": "str",
        "item_type": "str",
        "label": "评估数据集",
        "require": 0,
        "choice": [],
        "range": "",
        "default": "",
        "placeholder": "",
        "describe": "评估数据集",
        "editable": 1
      },
      "--feature_columns": {
        "type": "str",
        "item_type": "str",
        "label": "特征列，逗号分隔",
        "require": 0,
        "choice": [],
        "range": "",
        "default": "",
        "placeholder": "",
        "describe": "特征列，逗号分隔",
        "editable": 1
      },
      "--label_columns": {
        "type": "str",
        "item_type": "str",
        "label": "标签列",
        "require": 0,
        "choice": [],
        "range": "",
        "default": "",
        "placeholder": "",
        "describe": "标签列，逗号分割",
        "editable": 1
      },
      "--model_params": {
        "type": "json",
        "item_type": "str",
        "label": "模型参数",
        "require": 0,
        "choice": [],
        "range": "",
        "default": "",
        "placeholder": "",
        "describe": "模型参数",
        "editable": 1
      },
      "--save_model_dir": {
        "type": "str",
        "item_type": "str",
        "label": "模型保存目录",
        "require": 1,
        "choice": [],
        "range": "",
        "default": "",
        "placeholder": "",
        "describe": "模型保存目录",
        "editable": 1
      },
      "--inference_dataset": {
        "type": "str",
        "item_type": "str",
        "label": "推理数据集",
        "require": 0,
        "choice": [],
        "range": "",
        "default": "",
        "placeholder": "",
        "describe": "推理数据集",
        "editable": 1
      }
    }
}
```

