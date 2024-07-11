# pushgateway 使用教程
## 1、prometheus 数据统计

post数据格式
```
type:'prometheus'

metrics:
{
    'metric_name1':{
         'lables':['method', 'clientip'],
         'describe':'this is test',
         'exist_not_update_type':'clear',
         'exist_update_type':'update',
         'not_exist_update_type':'add',
         'push_deal_type':'clear',
         'data':[
            [['get','192.168.11.127'],4],
            [['get','192.168.12.49'],3],
            [['post','192.168.11.127'],5]
         ]
    },
    'metric_name2':{
    }
}
```
其中 

exist_not_update_type(已存在但不更新的数据),必选参数  
exist_update_type(已存在切更新的数据),必选参数  
not_exist_update_type(不存在但更新的数据),必选参数  
pull_finish_deal_type(数据被拉取以后的处理行为),必选参数  

可选参数
```
# update 覆盖原有值
# clear 删除属性
# keep 保留原状态
# add 属性的value累加
# reset 属性的值设置为0
```

python server内部存储结构
```
{
    'metric_name1':{
         'lables':['method', 'clientip'],
         'describe':'this is test',
         'exist_not_update_type':'clear',
         'exist_update_type':'update',
         'not_exist_update_type':'add',
         'data':{
                    ('get','192.168.11.127'):4,
                    ('get','192.168.12.49'):3,
                    ('post','192.168.11.127'):5
         }
    },
    'metric_name2':{
    }
}
```

## 2、报警推送代理

post访问接口`/{client}/webhook`

参数
sender_type：字符串。推送类型（目前支持wechat和username_group）
sender：字符串。推送者（TME_DataInfra或企业微信机器人key）
username：字符串。接收用户(逗号分隔多用户)(微信推送时为username，企业微信群推送时为空)
message: 推送字符串，如果有message字段，则仅推送message字段，否则除上面之外的所有字段会json序列化为message推送

