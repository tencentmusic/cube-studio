# xgboost 模板
描述：单机xgb训练，支持训练预测。

镜像：ccr.ccs.tencentyun.com/cube-studio/xgb_train_and_predict:v1  

环境变量：  
```bash
NO_RESOURCE_CHECK=true
TASK_RESOURCE_CPU=2
TASK_RESOURCE_MEMORY=4G
TASK_RESOURCE_GPU=0
```

启动参数：  
```bash
{
    "shell": {
        "--sep": {
            "type": "str",
            "item_type": "",
            "label": "分隔符",
            "require": 1,
            "choice": [
                "space",
                "TAB",
                ","
            ],
            "range": "",
            "default": ",",
            "placeholder": "",
            "describe": "分隔符",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--classifier_or_regressor": {
            "type": "str",
            "item_type": "",
            "label": "分类还是回归",
            "require": 1,
            "choice": [
                "classifier",
                "regressor"
            ],
            "range": "",
            "default": "classifier",
            "placeholder": "",
            "describe": "分类还是回归",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--params": {
            "type": "json",
            "item_type": "str",
            "label": "xgb参数",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "xgb参数, json格式",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--train_csv_file_path": {
            "type": "text",
            "item_type": "",
            "label": "训练集csv路径",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "训练集csv路径，首行是header，首列是label。为空则不做训练，尝试从model_load_path加载模型。",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--model_load_path": {
            "type": "text",
            "item_type": "",
            "label": "模型加载路径",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型加载路径。为空则不加载。",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--predict_csv_file_path": {
            "type": "text",
            "item_type": "",
            "label": "预测数据集csv路径",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "预测数据集csv路径，格式和训练集一致，顺序保持一致，没有label列。为空则不做predict。",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--predict_result_path": {
            "type": "text",
            "item_type": "",
            "label": "预测结果保存路径",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "预测结果保存路径，为空则不做predict。",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--model_save_path": {
            "type": "text",
            "item_type": "",
            "label": "模型文件保存路径",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型文件保存路径。为空则不保存模型。",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        },
        "--eval_result_path": {
            "type": "text",
            "item_type": "",
            "label": "模型评估报告保存路径",
            "require": 1,
            "choice": [],
            "range": "",
            "default": "",
            "placeholder": "",
            "describe": "模型评估报告保存路径。默认为空，想看模型评估报告就填。",
            "editable": 1,
            "condition": "",
            "sub_args": {}
        }
    }
}
```
