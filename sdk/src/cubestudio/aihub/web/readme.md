
# /info
获取基础元数据
```bash
"name": self.model.name[0],
"label": self.model.label[0],
"describe": self.model.description[0],
"field": self.model.field[0],
"scenes": self.model.scenes[0],
"status": self.model.status[0],
"version": self.model.version[0],
"doc": self.model.doc[0],
"pic": self.model.pic[0],
"web_example":self.web_examples,
"web_inputs": self.web_examples
```
其中inputs 类型包含

`('text', 'image', 'video', 'stream', 'text_select', 'image_select')`
```bash
self.type=type
self.name=name
self.label=label
self.describe=describe
self.choices=choices
self.default=default
self.validators=validators
```

# 推理接口
`/api/model/{self.name}/version/{self.version}/`

文本直接传，图片传base64编码，视频文件直接传文件，摄像头传流，多选传数组

返回结构
```bash
{
    "status": 0,
    "result": [{
      "text":'',
      "image":'base64编码'
    }],
    "message": ""
}
```



