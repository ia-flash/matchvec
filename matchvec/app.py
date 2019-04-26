import os
import json
import cv2
import logging
import numpy as np
#import pandas as pd
from flask import Flask, render_template, send_from_directory, request
from flask_cors import CORS
from process import predict_class, predict_objects

app = Flask(__name__)
cors = CORS(app)

@app.route('/<path:path>')
def build(path):
    return send_from_directory('../dist', path)

@app.route('/')
def status():
    return send_from_directory('../dist', "index.html")

@app.route('/preview')
def preview():
    return send_from_directory('../dist', "index.html")

@app.route('/sivnorm')
def sivnorm():
    return send_from_directory('../dist', "index.html")

@app.route('/api/object_detection',methods=['POST'])
def api_object_detection():
    images = request.files.getlist('image')
    res = list()
    for i in range(len(images)):
        nparr = np.frombuffer(images[i].read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        res.append(predict_objects(img))
    return json.dumps(res)

@app.route('/api/predict',methods=['POST'])
def api_predict():
    images = request.files.getlist('image')
    res = list()
    for i in range(len(images)):
        nparr = np.frombuffer(images[i].read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        res.append(predict_class(img))
    return json.dumps(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
