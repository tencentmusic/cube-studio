import base64
import json,io,time
import requests
from PIL import Image
host = 'http://192.168.3.100:20090/'
url = host.strip('/')+'/v1/models/electric-bicycle/versions/v20240701/predict'
print('请求地址：',url)

# # 电瓶车识别的示例
# with open("electric-bicycle.jpg", "rb") as image_file:
#     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
# # yolov8的请求示例
# # with open("client.jpg", "rb") as image_file:
# #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
# # 包装成json
# data = {
#     'image': encoded_string
# }
#
# # 发送post请求
# response = requests.post(url, data=json.dumps(data))
# print(response.content)
# # result = json.loads(response.json)
# # print(result)


# 批量请求
for pic_url in open('vision-electric-bicycle.txt',mode='r').readlines():
    res = requests.get(pic_url.strip())
    content = res.content
    encoded_string = base64.b64encode(content).decode('utf-8')
    data = {
        'image': encoded_string
    }
    response = requests.post(url, data=json.dumps(data))
    print(response.content)
