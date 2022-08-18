import requests

from __future__ import print_function

import base64
import io
import json

import numpy as np
from PIL import Image    # pip install Pillow
import requests

headers={
    "Content-Type": "application/json"
}

# onnxruntime
SERVER_URL = 'http://onnxruntime-resnet50-202112281143/v1/models/resnet50'
IMAGE_PATH = 'smallcat.jpg'
files = {'data': open(IMAGE_PATH, 'rb')}
response = requests.post(SERVER_URL, data=files,headers=headers)
print(response.json())
# print(response.content)
print(response.status_code)
response.raise_for_status()




