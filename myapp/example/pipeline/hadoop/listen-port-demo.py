import os
os.system('pip3 install -U pip && pip install -U flask')

from flask import Flask

app = Flask(__name__)

ip=os.getenv('K8S_HOST_IP','')
port=os.getenv('PORT1','')
print(ip,port)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)