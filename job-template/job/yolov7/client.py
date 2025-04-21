import base64
import json,io,time
import requests
from PIL import Image
host = 'http://192.168.3.100:20070/'
url = host.strip('/')+'/v1/models/yolov7/versions/20240101/predict'
print('请求地址：',url)
# 读取图片并编码成base64
with open("client.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# 包装成json
data = {
    'image': encoded_string
}

# 发送post请求
response = requests.post(url, data=json.dumps(data))
result = json.loads(response.text)
print(result)
image_data = result['image']
# 解码图像
image_bytes = base64.b64decode(image_data)
image = Image.open(io.BytesIO(image_bytes))
save_path = f'result.jpg'
image.save(save_path)
print('结果保存到result.jpg')
