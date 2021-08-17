from flask import render_template
from flask import Flask,request
import logging
app = Flask(__name__)




@app.route('/')
def index():
    logging.info('>>>>>>>>>>>>>')
    ip = request.remote_addr
    print(ip)
    logging.info(ip)
    logging.info('<<<<<<<<<<<<<')
    return render_template("./index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port='80')