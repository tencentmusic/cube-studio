
import sys
import os

import traceback
import argparse
import base64
import logging
import time,datetime
import json
import requests
from flask import redirect
import os
from os.path import splitext, basename
import time
import numpy as np
import datetime
import logging
import flask
import werkzeug
import optparse
import cv2
from flask import jsonify,request
from PIL import Image,ImageFont
from PIL import ImageDraw
import urllib
from PIL import Image

import pysnooper
from flask import Flask

app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')

UPLOAD_FOLDER = "UPLOAD_FOLDER"
# myfont = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 20)


from aihub import start

@app.route('/api/v1.0/model',methods=['GET','POST'])
# @pysnooper.snoop(watch_explode=('all_back_image',))
def classify_rest():
    try:
        data = request.json
        txt = data.get('txt','姓')
        if not txt:
            txt='姓'
        image_decode = base64.b64decode(data['image_data'])
        image_path = os.path.join(UPLOAD_FOLDER, str(datetime.datetime.now()) + ".jpg")
        nparr = np.fromstring(image_decode, np.uint8)
        # 从nparr中读取数据，并把数据转换(解码)成图像格式
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(image_path, img_np)

        logging.info('Saving to %s.', image_path)

        all_back_image = start(
            img_path=image_path,
            text=txt
        )

        os.remove(image_path)
        return jsonify({
            "status": 0,
            "result": all_back_image,
            "message":""
        })

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return jsonify(val = 'Cannot open uploaded image.')


@app.route('/')
def hello():
    return redirect('/static/index.html')

if __name__=='__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0',debug=True,port='8080')

