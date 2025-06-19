
## 推理接口

通过该接口将媒体内容发送给inference接口，进行模型推理，获取推理后的结果

- **URL**：/v1/models/MODEL_NAME/versions/MODEL_VERSION/predict
- **Method**：POST
- **Content-Type**： application/json
- **需要登录**：否

## headers
```
{
    'Content-Type': 'application/json'
}
```
## 输入
```bash
{
    "image":base64.b64encode(open(image_path, "rb").read()).decode('utf-8')
}
```

## 输出

**状态码**：200 OK  

```
{
    "labels":[],
    "scores":[],
    "xywhs":[],
    "orig_shape":[orig_width,orig_height]
}
```

## 其他推理类型示例接口

通过该接口将媒体内容发送给inference接口，进行模型推理，获取推理后的结果

- **URL**：/predict
- **Method**：POST
- **Content-Type**： application/json
- **需要登录**：否

## headers
```
{
    'Content-Type': 'application/json'
}
```
## 输入
```bash
{
    "image":base64.b64encode(open(image_path, "rb").read()).decode('utf-8')
}
```

## 输出

**状态码**：200 OK  

```
{
    "results":[],
}
```

# 自动化标注接口
```
/labelstudio
```