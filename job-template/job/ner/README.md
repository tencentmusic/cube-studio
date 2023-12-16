# ner 模板

## 模版简介
  命名实体识别（NER）是一种自然语言处理技术，可以自动扫描整篇文章，提取文本中的一些基本实体，并将它们分类到预定义的类别中。
  举一个直观一点的例子，比如你对手机的语音助手说，“提醒我明天下午7点开会”，或者“明天北京海淀区的天气怎么样”，然后它就会根据你的指令来做出相应的执行。
  主要应用于：搜索和推荐引擎、自动聊天机器人、内容分析、消费者洞察

## 准备数据集

   数据集 可以通过[链接](https://docker-76009.sz.gfp.tencent-cloud.com/github/cube-studio/pipeline/NER.zip)下载

   把数据拷贝到 note book 中 `/mnt/admin/NER` 目录（可以直接拖拽进去）,包含NER/zdata/resume_BIO.txt和NER/zdata/people_daily_BIO.txt

## 注册模板

```json
{
   "参数分组1": {
       "--model": {
           "type": "str",
           "item_type": "str",
           "label": "训练的基础模型名称，这里固定为: BiLSTM_CRF",
           "require": 1,
           "choice": [],
           "range": "",
           "default": "BiLSTM_CRF",
           "placeholder": "",
           "describe": "训练的基础模型名称，这里固定为: BiLSTM_CRF",
           "editable": 1
       },
       "--path": {
           "type": "str",
           "item_type": "str",
           "label": "训练数据存放目录",
           "require": 1,
           "choice": [],
           "range": "",
           "default": "/mnt/admin/NER/zdata/",
           "placeholder": "",
           "describe": "训练数据存放目录",
           "editable": 1
       },
       "--filename": {
           "type": "str",
           "item_type": "str",
           "label": "数据集的名字",
           "require": 1,
           "choice": [
               "resume_BIO.txt",
               "people_daily_BIO.txt"
           ],
           "range": "",
           "default": "resume_BIO.txt",
           "placeholder": "",
           "describe": " 数据集的名字",
           "editable": 1
       },
       "--epochs": {
           "type": "str",
           "item_type": "str",
           "label": "训练的次数，次数越大效果越好，建议 5 以上",
           "require": 1,
           "choice": [],
           "range": "",
           "default": "5",
           "placeholder": "",
           "describe": "训练的次数，次数越大效果越好，建议 5 以上",
           "editable": 1
       },
       
       "-pp": {
           "type": "str",
           "item_type": "str",
           "label": "模型保存地址",
           "require": 1,
           "choice": [],
           "range": "",
           "default": "/mnt/admin/model.pkl",
           "placeholder": "",
           "describe": "模型保存地址",
           "editable": 1
       }
   }
}
```



