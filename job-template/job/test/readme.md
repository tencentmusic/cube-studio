任务模板的启动参数配置
```
{
  "训练参数": {
    "--input_csv": {
      "type": "str",
      "item_type": "str",
      "label": "输入地址",
      "require": 1,
      "choice": [],
      "range": "",
      "default": "",
      "placeholder": "",
      "describe": "输入csv地址",
      "editable": 1
    },
    "--max_depth": {
      "type": "str",
      "item_type": "str",
      "label": "超参数1",
      "require": 1,
      "choice": [],
      "range": "",
      "default": "5",
      "placeholder": "",
      "describe": "超参数1",
      "editable": 1
    },
    "--min_samples_split": {
      "type": "str",
      "item_type": "str",
      "label": "超参数2",
      "require": 1,
      "choice": [],
      "range": "",
      "default": "10",
      "placeholder": "",
      "describe": "超参数2",
      "editable": 1
    },
    "--min_samples_leaf": {
      "type": "str",
      "item_type": "str",
      "label": "超参数3",
      "require": 1,
      "choice": [],
      "range": "",
      "default": "5",
      "placeholder": "",
      "describe": "超参数3",
      "editable": 1
    },
    "--output_dir": {
      "type": "str",
      "item_type": "str",
      "label": "保存目录",
      "require": 1,
      "choice": [],
      "range": "",
      "default": "",
      "placeholder": "",
      "describe": "保存目录",
      "editable": 1
    }
  }
}
```